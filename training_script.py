"""
Training script with optional FSDP (Fully Sharded Data Parallel) support for multi-GPU training.
Run with torchrun, e.g.:
torchrun --nproc_per_node=4 training_script.py --use-fsdp
"""

import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from utils import (
    calculate_total_train_params,
    save_to_checkpoint,
    setup_data,
)
from setup_model import get_models
from datasetlite import DataLoaderLite

from torch.distributed import is_initialized as dist_is_initialized
import torch.distributed as dist

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
except Exception:
    FSDP = None  # FSDP not available (e.g., MPS / CPU only)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--accumulation-steps', type=int, default=1)
    p.add_argument('--caption-len', type=int, default=20)
    p.add_argument('--max-samples', type=int, default=1000)
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default='bf16')
    p.add_argument('--use-fsdp', action='store_true', help='Enable FSDP sharding')
    p.add_argument('--fsdp-min-params', type=int, default=10_000_000, help='Auto wrap min params')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    p.add_argument('--checkpoint-path', type=str, default='checkpoint_fsdp.pt', help='Checkpoint file path')
    p.add_argument('--save-every', type=int, default=1000, help='Global steps interval for checkpoint saving')
    return p.parse_args()


def init_distributed():
    if dist_is_initialized():
        return
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Single-process fallback
        rank = 0
        world_size = 1
        local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = 'nccl'
    else:
        backend = 'gloo'

    print (f"RANK {rank} world_Size {world_size}")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def get_device(local_rank):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{local_rank}')
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def build_mixed_precision(precision):
    if FSDP is None:
        return None
    if precision == 'fp16':
        return MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
    if precision == 'bf16':
        return MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    return None


def fsdp_wrap(model, mp_cfg, min_params):
    if FSDP is None:
        return model
    auto_wrap = size_based_auto_wrap_policy
    wrapped = FSDP(
        model,
        auto_wrap_policy=lambda m, recurse, nonwrapped: auto_wrap(
            m, recurse, nonwrapped, min_num_params=min_params
        ),
        mixed_precision=mp_cfg,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
    )
    return wrapped


def is_rank0():
    return (not dist_is_initialized()) or dist.get_rank() == 0


def save_fsdp_checkpoint(encoder_model, decoder_model, optimizer, scheduler, scaler, epoch, global_step, config, path):
    """Save checkpoint. Works for FSDP or non-FSDP models. Only rank0 writes to disk."""
    if not is_rank0():
        return
    ckpt = {
        'epoch': epoch,
        'global_step': global_step,
        'config': {
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'lr': config.lr,
            'accumulation_steps': config.accumulation_steps,
            'caption_len': config.caption_len,
        },
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
    }
    # Retrieve full (unsharded) state dict if FSDP
    if FSDP is not None and isinstance(encoder_model, FSDP):
        with FSDP.state_dict_type(encoder_model, state_dict_type='full_state_dict', rank0_only=True), \
             FSDP.state_dict_type(decoder_model, state_dict_type='full_state_dict', rank0_only=True):
            ckpt['encoder_state'] = encoder_model.state_dict()
            ckpt['decoder_state'] = decoder_model.state_dict()
    else:
        ckpt['encoder_state'] = encoder_model.state_dict()
        ckpt['decoder_state'] = decoder_model.state_dict()
    tmp_path = path + '.tmp'
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    if is_rank0():
        print(f'[checkpoint] Saved to {path} (epoch {epoch}, step {global_step})')


def load_fsdp_checkpoint(encoder_model, decoder_model, optimizer, scheduler, scaler, path, device):
    """Load checkpoint (rank0 loads and broadcasts if distributed). Returns epoch, global_step."""
    if not os.path.exists(path):
        if is_rank0():
            print(f'[checkpoint] No checkpoint found at {path}, starting fresh.')
        return 0, 0
    # Rank0 loads
    if is_rank0():
        ckpt = torch.load(path, map_location=device)
    else:
        ckpt = None
    # Broadcast availability
    if dist_is_initialized():
        has_ckpt = torch.tensor([1 if ckpt is not None else 0], device=device)
        dist.broadcast(has_ckpt, src=0)
        if has_ckpt.item() == 0:
            return 0, 0
        # Broadcast state dict (simplified: using barrier + rank0 load; other ranks re-load from disk)
        dist.barrier()
        if not is_rank0():
            ckpt = torch.load(path, map_location=device)
    # Load model states
    if FSDP is not None and isinstance(encoder_model, FSDP):
        with FSDP.state_dict_type(encoder_model, state_dict_type='full_state_dict', rank0_only=True), \
             FSDP.state_dict_type(decoder_model, state_dict_type='full_state_dict', rank0_only=True):
            encoder_model.load_state_dict(ckpt['encoder_state'])
            decoder_model.load_state_dict(ckpt['decoder_state'])
    else:
        encoder_model.load_state_dict(ckpt['encoder_state'])
        decoder_model.load_state_dict(ckpt['decoder_state'])
    optimizer.load_state_dict(ckpt['optimizer'])
    try:
        scheduler.load_state_dict(ckpt['scheduler'])
    except Exception:
        pass
    try:
        scaler.load_state_dict(ckpt['scaler'])
    except Exception:
        pass
    epoch = ckpt.get('epoch', 0)
    global_step = ckpt.get('global_step', 0)
    if is_rank0():
        print(f'[checkpoint] Resumed from {path} at epoch {epoch}, step {global_step}')
    return epoch, global_step


scaler = GradScaler()
loss_list = []


class MultiModelTrainer():

    def __init__(self, encoder_model, decoder_model, train_dataloader, val_dataloader, epochs, device, config, optimizer, scheduler):
        self.train_dataloader = train_dataloader
        self.epochs = epochs
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_steps = len(self.train_dataloader) * self.config.epochs
        self.trainable_params = list([p for p in self.encoder_model.parameters() if p.requires_grad]) + \
                                list([p for p in self.decoder_model.parameters() if p.requires_grad])
        self.running_total_loss = 0.0


    
    def get_total_steps(self):
        return len(self.train_dataloader) * self.config.epochs

    def should_stop(self, loss_list):
        last_ten_loss = loss_list[-50*4:]
        threshold = 0.5
        if len(last_ten_loss)==50*4 and len(loss_list)>=50*4:
            diffs = np.diff(last_ten_loss)
            step_trends = []
            for d in diffs:
                if d > threshold:
                    step_trends.append("increasing")
                elif d < -threshold:
                    step_trends.append("decreasing")
                else:
                    step_trends.append("steady")

            if all(t == "steady" for t in step_trends):
                return True 
            else:
                print ("Trend", step_trends)
        return False 


    
    def eval(self, epoch):
        self.decoder_model.eval()
        self.encoder_model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for val_batch in self.val_dataloader:
                # Unpack batch dict
                image_tensor = val_batch["image"].to(self.device)
                caption_tensor = val_batch["input_ids"].to(self.device)
                attention_mask = val_batch["attention_mask"].to(self.device)
                bboxes = val_batch["bboxes"].to(self.device)
                class_labels = val_batch["class_labels"].to(self.device)
                objectness = val_batch["objectness"].to(self.device)

                with torch.autocast(self.device.type if self.device.type != 'cpu' else 'cpu', enabled=True, dtype=torch.bfloat16):
                    x_embed = self.encoder_model(image_tensor)
                    # decoder_model should return caption_loss and detection_loss
                    #img_features, captions_tensor, attention_mask=None, bbox_targets=None, class_targets=None, objectness_targets=None, 
                    logits, bbox_preds, objectness_pred, class_pred, loss_list = self.decoder_model(
                        x_embed, 
                        caption_tensor, 
                        attention_mask,
                        bbox_targets=bboxes,
                        class_targets=class_labels,
                        objectness_targets=objectness,
                        mode="train"
                    )
                    total_loss, lm_loss, loss_bbox, loss_giou, loss_ce, objectness_loss = loss_list[:]
                    # combine losses
                    total_val_loss = total_loss

                val_loss += total_val_loss.item()
                count += 1
                if count > 2: break  # quick validation
        val_loss /= max(count, 1)
        # All-reduce across ranks if distributed
        if dist_is_initialized():
            loss_tensor = torch.tensor([val_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = (loss_tensor / dist.get_world_size()).item()
        self.decoder_model.train()
        self.encoder_model.train()
        if is_rank0():
            print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")
        return val_loss


    def train(self):
        for epoch in range(self.epochs):
            for step, batch in enumerate(self.train_dataloader):
                image_tensor = batch["image"].to(self.device)
                caption_tensor = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bboxes = batch["bboxes"].to(self.device)
                class_labels = batch["class_labels"].to(self.device)
                objectness = batch["objectness"].to(self.device)

                global_step = epoch * len(self.train_dataloader) + step + 1

                with torch.autocast(self.device.type if self.device.type != 'cpu' else 'cpu', enabled=True, dtype=torch.bfloat16):
                    x_embed = self.encoder_model(image_tensor)
                    logits, bbox_preds, objectness_pred, class_pred, loss_list = self.decoder_model(
                        x_embed,
                        caption_tensor,
                        attention_mask,
                        bbox_targets=bboxes,
                        class_targets=class_labels,
                        objectness_targets=objectness
                    )
                    loss_1, lm_loss, loss_bbox, loss_giou, loss_ce, objectness_loss = loss_list[:]

                    loss = loss_1 / self.config.accumulation_steps

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                if FSDP is not None and isinstance(self.decoder_model, FSDP):
                    self.decoder_model.clip_grad_norm_(max_norm=5.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=5.0)

                if (step + 1) % self.config.accumulation_steps == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                self.running_total_loss += loss.item() * self.config.accumulation_steps

                if global_step % 100 == 0:
                    val_loss = self.eval(epoch)
                    if is_rank0():
                        loss_list.append(val_loss)
                    stop_flag = torch.tensor([0], device=self.device)
                    if is_rank0() and self.should_stop(loss_list):
                        stop_flag[0] = 1
                    if dist_is_initialized():
                        dist.broadcast(stop_flag, src=0)
                    if stop_flag.item() == 1:
                        if is_rank0():
                            print('Early stop triggered.')
                        return

                if is_rank0():
                    if global_step % self.config.save_every == 0:
                        save_fsdp_checkpoint(
                            self.encoder_model,
                            self.decoder_model,
                            self.optimizer,
                            self.scheduler,
                            scaler,
                            epoch,
                            global_step,
                            self.config,
                            self.config.checkpoint_path
                        )
                    if global_step % 100 == 0:
                        elapsed = time.time() - start_time
                        steps_per_sec = max(global_step / elapsed, 1e-9)
                        remaining_steps = self.total_steps - global_step
                        est_remaining = remaining_steps / steps_per_sec
                        est_total = self.total_steps / steps_per_sec
                        mem_str = ''
                        if torch.cuda.is_available():
                            mem_str = f"CUDA Mem {torch.cuda.memory_allocated()/1e9:.2f}GB/{torch.cuda.memory_reserved()/1e9:.2f}GB"
                        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                            try:
                                mem_str = f"MPS Mem {torch.mps.current_allocated_memory()/1e9:.2f}GB/{torch.mps.driver_allocated_memory()/1e9:.2f}GB"
                            except Exception:
                                mem_str = 'MPS Mem N/A'
                        print(
                            f"epoch {epoch+1}/{self.config.epochs} step {step}/{len(self.train_dataloader)} "
                            f"Loss: {loss.item()*self.config.accumulation_steps:.4f} | "
                            f"Elapsed: {elapsed/60:.2f}m | ETA: {est_remaining/60:.2f}m | Total est: {est_total/60:.2f}m | "
                            f"{mem_str} | PPL {math.exp(min(loss.item()*self.config.accumulation_steps, 20)):.2f}"
                        )

            # end epoch cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            import gc; gc.collect()


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = get_device(local_rank)
    torch.manual_seed(args.seed + (rank if dist_is_initialized() else 0))

    # Load models/config
    TrainingConfig, encoder_model, decoder_model, pad_token_id, tokenizer, extras_dict = get_models()
    TrainingConfig.batch_size = args.batch_size
    TrainingConfig.epochs = args.epochs
    TrainingConfig.lr = args.lr
    TrainingConfig.accumulation_steps = args.accumulation_steps
    TrainingConfig.caption_len = args.caption_len
    TrainingConfig.save_every = args.save_every
    TrainingConfig.checkpoint_path = args.checkpoint_path


    print (f"\n\t BATCH SIZE { TrainingConfig.batch_size}")

    # Data setup (each rank gets its own sampler)
    train_cap, val_cap, id_to_name, name_to_id = setup_data(args.max_samples, val_split=args.val_split)
    train_dataset = DataLoaderLite(train_cap, caption_length=args.caption_len, num_img_tokens=64, tokenizer=tokenizer)
    val_dataset = DataLoaderLite(val_cap, caption_length=args.caption_len, num_img_tokens=64, tokenizer=tokenizer)

    if dist_is_initialized():
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=train_sampler is None)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False)

    # Optimizer / Scheduler setup
    all_params = calculate_total_train_params(encoder_model, decoder_model)
    optimizer = torch.optim.AdamW(all_params, lr=TrainingConfig.lr)
    total_steps = len(train_dataloader) * TrainingConfig.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps/TrainingConfig.accumulation_steps, eta_min=1e-6)

    # FSDP wrap if requested
    mp_cfg = build_mixed_precision(args.precision)
    if args.use_fsdp and FSDP is not None and torch.cuda.is_available():
        encoder_model = fsdp_wrap(encoder_model, mp_cfg, args.fsdp_min_params)
        decoder_model = fsdp_wrap(decoder_model, mp_cfg, args.fsdp_min_params)
        if is_rank0():
            print('FSDP enabled.')
    else:
        if args.use_fsdp and FSDP is None and is_rank0():
            print('FSDP requested but not available; continuing without it.')

    # Move models
    encoder_model.to(device)
    decoder_model.to(device)

    if is_rank0():
        print(f"Trainable params: {sum(p.numel() for p in all_params if p.requires_grad)/1e6:.2f}M")

    global start_time
    start_time = time.time()
    trainer = MultiModelTrainer(encoder_model, decoder_model, train_dataloader, val_dataloader, TrainingConfig.epochs, device, TrainingConfig, optimizer, scheduler)
    # Resume logic (after wrapping and optimizer creation)
    start_epoch = 0
    start_global_step = 0
    if args.resume:
        start_epoch, start_global_step = load_fsdp_checkpoint(encoder_model, decoder_model, optimizer, scheduler, scaler, args.checkpoint_path, device)
        # Adjust scheduler position if resuming mid-training
        if start_global_step > 0:
            # advance scheduler steps already consumed
            steps_to_advance = start_global_step // TrainingConfig.accumulation_steps
            for _ in range(steps_to_advance):
                scheduler.step()
        # Adjust epochs to continue remaining
        if start_epoch > 0 and start_epoch < TrainingConfig.epochs:
            trainer.epochs = TrainingConfig.epochs  # keep total epochs; internal loop will continue
            # Could skip completed epochs by setting a starting offset, but for simplicity we rely on global_step
        if is_rank0():
            print(f'Resuming training: start_epoch={start_epoch}, start_global_step={start_global_step}')
    trainer.train()

    if dist_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()