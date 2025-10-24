from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor
from transformers.models.grounding_dino import GroundingDinoForObjectDetection
import os 
from dotenv import load_dotenv
from transformers import GroundingDinoConfig
import torch 

load_dotenv()

device = torch.device("mps")


model_id = "IDEA-Research/grounding-dino-tiny"
config = GroundingDinoConfig()


processor = AutoProcessor.from_pretrained(model_id)
model = GroundingDinoForObjectDetection.from_pretrained(model_id)
model = model.to(device)
tokenizer = processor.tokenizer

# Ensure pad token exists (some GroundingDINO tokenizers may lack it) so manual padding works for multi-text batches.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    try:
        model.resize_token_embeddings(len(tokenizer))
        print("[Init] Added pad token to tokenizer and resized embeddings")
    except Exception as e:
        print(f"[Init] Failed to resize embeddings after adding pad token: {e}")



if not hasattr(torch.nn.functional, "_original_grid_sample"):
    torch.nn.functional._original_grid_sample = torch.nn.functional.grid_sample

    def safe_grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
        if input.device.type == "mps":
            # Move to CPU + float32
            input_cpu = input.to("cpu", dtype=torch.float32)
            grid_cpu = grid.to("cpu", dtype=torch.float32)
            with torch.no_grad():
                output_cpu = torch.nn.functional._original_grid_sample(
                    input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode, align_corners=align_corners
                )
            return output_cpu.to("mps", dtype=input.dtype)
        else:
            return torch.nn.functional._original_grid_sample(
                input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
            )

torch.nn.functional.grid_sample = safe_grid_sample

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import re 


UI_VOCAB = [
    "button", "icon", "text", "image", "input", "checkbox", "link",
    "menu item", "banner", "avatar", "logo", "label", "switch",
    "tab", "card", "popup", "dropdown", "textfield", "container"
]

UI_TEXT_MAP = {
    "image": ["picture", "photo", "avatar", "logo", "icon"],
    "button": ["button", "tap", "click", "submit"],
    "text": ["text", "label"],
    "input": ["input", "field", "search", "textbox"],
}


class UIElementClassifier:
    def __init__(self, vocab):
        self.vocab = vocab
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vocab_emb = self.model.encode(vocab, normalize_embeddings=True)
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")



    def build_positive_map(self, text: str, phrases: list[str]) -> torch.Tensor:
        # Tokenize text using DINO processor to get offsets
        encoding = self.processor(
            text=[text],
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
        )

        offsets = encoding["offset_mapping"][0]  # [seq_len, 2]
        positive_map = torch.zeros((len(phrases), offsets.shape[0]), dtype=torch.bool)

        text_lower = text.lower()

        for i, phrase in enumerate(phrases):
            # Map to candidate synonyms if available
            candidates = UI_TEXT_MAP.get(phrase.lower().strip(), [phrase])
            candidates+=[phrase]

            for cand in candidates:
                for match in re.finditer(re.escape(cand), text_lower):
                    start, end = match.span()
                    # Mark all tokens that overlap the match
                    for j, (s_tok, e_tok) in enumerate(offsets):
                        if e_tok > start and s_tok < end:
                            positive_map[i, j] = True

        return positive_map

        
    def classify(self, text):
        """Return best-matching UI term for the referring expression."""
        query_emb = self.model.encode(text, normalize_embeddings=True)
        sim = util.cos_sim(query_emb, self.vocab_emb)[0]
        idx = torch.argmax(sim).item()
        return self.vocab[idx], sim[idx].item()



from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pandas as pd
import os 
import ast 
import numpy as np 
import torch
from sentence_transformers import SentenceTransformer, util
import json


UI_VOCAB = [
    "button", "icon", "text", "image", "input", "checkbox", "link",
    "menu item", "banner", "avatar", "logo", "label", "switch",
    "tab", "card", "popup", "dropdown", "textfield", "container"
]

UI_TEXT_MAP = {
    "image": ["picture", "photo", "avatar", "logo", "icon"],
    "button": ["button", "tap", "click", "submit"],
    "text": ["text", "label"],
    "input": ["input", "field", "search", "textbox"],
}



class OSAtlasRefDataset(Dataset):
    """
    Converts OS-Atlas GUI grounding dataset to GroundingDINO format.
    Each image may contain multiple referring expressions and bounding boxes.
    """

    def __init__(self, json_path, image_root, tokenizer, 
                 split="train", auto_add_tokens=False):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.auto_add_tokens = auto_add_tokens
        self.image_root = image_root

        # Load raw JSON
        with open(json_path, "r") as f:
            self.data = json.load(f)

        duplicate_dict = {}

        # Flatten image + elements into one dataframe
        records = []
        for item in self.data:
            img_path = os.path.join(image_root, item["img_filename"])

            # if img_path!='/Users/preetamverma/Downloads/screenshots/67892.jpg':
            #     continue

            for el in item["elements"]:

                key = (img_path,  tuple(el["bbox"]))
                if key in duplicate_dict:
                    continue
                duplicate_dict[key]=True

                records.append({
                    "img_path": img_path,
                    "instruction": el["instruction"].strip(),
                    "bbox": el["bbox"]  # assumed xyxy (absolute or normalized)
                })
        self.df = pd.DataFrame(records)
        print(f" Loaded {len(self.df)} referring expressions from {len(self.data)} images")

        self.ui_classifier = UIElementClassifier(UI_VOCAB)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _normalize_xyxy(bbox, width, height):
        """Ensure bbox is normalized xyxy in [0,1]. Accept list/tuple of 4 numbers."""
        x0, y0, x1, y1 = bbox
        # If values look already normalized keep them (heuristic: all <= 1.2)
        if max(x0, y0, x1, y1) > 1.2:  # treat as pixel coords
            x0 /= width
            x1 /= width
            y0 /= height
            y1 /= height
        # Clamp to [0,1]
        x0 = min(max(x0, 0.0), 1.0)
        y0 = min(max(y0, 0.0), 1.0)
        x1 = min(max(x1, 0.0), 1.0)
        y1 = min(max(y1, 0.0), 1.0)
        # Fix inverted boxes if any
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        return [x0, y0, x1, y1]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]

        #print (f"Processing {img_path}")

        text = row["instruction"]
        bbox = row["bbox"]  # expected xyxy

        # Load image
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # Normalize bbox to [0,1] xyxy
        bbox = self._normalize_xyxy(bbox, W, H)

        ui_label, score = self.ui_classifier.classify(text.lower())
        phrases = [ui_label]
        bbox_final = [bbox] * len(phrases)  # one box per phrase currently

        positive_map = self.ui_classifier.build_positive_map(text, phrases)
        if not torch.any(positive_map):
            # Skip logic handled in collate; still return placeholder
            pass

        class_labels = [index for index in range(len(phrases))]

        return {
            "image": image,
            "text": text,
            "bbox": bbox_final,          # list of normalized xyxy
            "class_labels": class_labels,
            "positive_map": positive_map,
            "phrases": phrases,
        }


dataset = OSAtlasRefDataset(
    json_path="/Users/preetamverma/Downloads/uibert_raw.json",  
    image_root="/Users/preetamverma/Downloads",      
    tokenizer=tokenizer
)

import random 
from PIL import Image

UNIFORM_SIZE = (512, 512)  # width, height


def visualize_boxes(image, boxes, labels, scores):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = box
        width, height = x1 - x0, y1 - y0
        rect = patches.Rectangle((x0, y0), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0, f"{label}: {score:.2f}", color='white', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5))

    plt.show()


def _xyxy_to_cxcywh_norm(xyxy: torch.Tensor):
    x0, y0, x1, y1 = xyxy.unbind(-1)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0).clamp(min=1e-6)
    h = (y1 - y0).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def _resize_all(images):
    resized = []
    for img in images:
        if img.size != UNIFORM_SIZE:
            resized.append(img.resize(UNIFORM_SIZE))
        else:
            resized.append(img)
    return resized


def collate_fn(batch):
    # Per-sample tokenization: processor collapses multiple texts, so we tokenize texts individually.
    filtered = [item for item in batch if item["text"].strip() and torch.any(item["positive_map"])]
    if not filtered:
        # print(" Collated batch of 0 samples")
        return None

    per_images = []
    per_text_encodings = []
    labels = []
    max_seq_len = 0

    for sample_idx, item in enumerate(filtered):
        try:
            txt = item["text"]
            img = item["image"]
            text_enc = tokenizer(txt, return_tensors="pt", truncation=True)
            seq_len = text_enc["input_ids"].shape[-1]
            max_seq_len = max(max_seq_len, seq_len)

            class_labels = torch.as_tensor(item["class_labels"], dtype=torch.long)
            box_tensors = [torch.tensor(b, dtype=torch.float32) for b in item["bbox"]]
            boxes = torch.stack(box_tensors, dim=0)
            boxes_cxcywh = _xyxy_to_cxcywh_norm(boxes)
            pos_map = item["positive_map"].to(torch.bool)

            if boxes_cxcywh.ndim != 2 or boxes_cxcywh.shape[-1] != 4:
                raise RuntimeError(f"Invalid boxes shape {boxes_cxcywh.shape}")
            if class_labels.shape[0] != boxes_cxcywh.shape[0]:
                raise RuntimeError("Mismatch class_labels vs boxes count")
            if pos_map.shape[0] != boxes_cxcywh.shape[0]:
                raise RuntimeError("Mismatch positive_map rows vs boxes count")

            per_images.append(img)
            per_text_encodings.append(text_enc)
            labels.append({
                "class_labels": class_labels.to(device),
                "boxes": boxes_cxcywh.to(device),
                "positive_map": pos_map
            })
        except Exception as e:
            print(f"  Drop sample {sample_idx}: {e}")

    if not labels:
        # print(" Collated batch ended empty after drops")
        return None

    # Pad text encodings
    batch_input_ids = []
    batch_attn = []
    for te in per_text_encodings:
        ids = te["input_ids"][0]
        attn = te["attention_mask"][0]
        pad_len = max_seq_len - ids.shape[0]
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad_ids])
            attn = torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)])
        batch_input_ids.append(ids)
        batch_attn.append(attn)
    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_attn = torch.stack(batch_attn, dim=0)

    # Pad positive_map to text length
    for l in labels:
        pm = l["positive_map"]
        if pm.shape[-1] < max_seq_len:
            pad = max_seq_len - pm.shape[-1]
            pm = torch.cat([pm, torch.zeros((pm.shape[0], pad), dtype=torch.bool)], dim=-1)
        elif pm.shape[-1] > max_seq_len:
            pm = pm[:, :max_seq_len]
        l["positive_map"] = pm.to(device)

    # Image processing (pixel_values)
    image_enc = processor.image_processor(per_images, return_tensors="pt")
    encodings = {
        "pixel_values": image_enc["pixel_values"].to(device),
        "input_ids": batch_input_ids.to(device),
        "attention_mask": batch_attn.to(device),
    }

    if encodings["input_ids"].shape[0] != len(labels):
        raise RuntimeError(f"Final mismatch: input_ids {encodings['input_ids'].shape[0]} vs labels {len(labels)}")

    # print(f" Collated batch size {len(labels)} | seq_len {max_seq_len}")
    return encodings, labels, per_images


    #### TRAINING LOOP #####

import os
import numpy as np 
from utils import calculate_total_train_params, save_to_checkpoint
import torch.nn as nn 
from typing import ClassVar
from pydantic import BaseModel
import torch.nn.functional as F
import time 
import math


class TrainingConfig(BaseModel):
    batch_size: ClassVar[int] = 2
    steps: ClassVar[int] = 0
    epochs: ClassVar[int] = 1
    lr: ClassVar[float] = 5e-4
    accumulation_steps: ClassVar[int] = 2
    save_every : ClassVar[int] = 1000
    checkpoint_path : ClassVar[str] = "checkpoint.pth"


train_dataloader = DataLoader(dataset, batch_size=TrainingConfig.batch_size, shuffle=True, collate_fn=collate_fn)

##### Setup Training #####

all_params = calculate_total_train_params(model)

total_steps = len(train_dataloader)  * TrainingConfig.epochs


print (f"Trainable parameters in encoder model: {sum(p.numel() for p in all_params if p.requires_grad)/1e6} M")

optimizer = torch.optim.AdamW(all_params, lr=TrainingConfig.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps/TrainingConfig.accumulation_steps, eta_min=1e-6)

start_time = time.time()
total_loss = 0 
best_val_loss = float("inf")
epochs_no_improve = 0
steps_no_improve = 0
patience_steps = 1
stop = False 
device = torch.device("mps")

l_epoch = 0 
l_loss = 0 
l_epoch =0
l_global_step = 0


N_EPOCHS = TrainingConfig.epochs - l_epoch

# print (f"PREVIOUS LOSS {l_loss} AT GLOBAL STEP {l_global_step} AT EPOCH {l_epoch}")

# Optional: disable autocast if instability persists
use_autocast = True
duplicate_dict = {}

for epoch in range(N_EPOCHS):

    for step, batch in enumerate(train_dataloader):

        global_step = epoch * len(train_dataloader) + step + 1 

        if batch is None:
            continue

        if global_step > 100: break

        # print (f"GLOBAL STEP {global_step}")

        enc , labels, images = batch

        enc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in enc.items()}

        # Debug targets before forward
        for li, l in enumerate(labels):
            if torch.any(torch.isnan(l["boxes"])) or torch.any(torch.isinf(l["boxes"])):
                print(f"[WARN] NaN/Inf in label boxes idx={li}", l["boxes"])
            if (l["boxes"] < 0).any() or (l["boxes"] > 1).any():
                print(f"[WARN] Out-of-range boxes idx={li}", l["boxes"])    
            if l["boxes"].ndim != 2 or l["boxes"].shape[-1] != 4:
                print(f"[WARN] Unexpected box shape {l['boxes'].shape}")

        amp_ctx = torch.autocast("mps", enabled=use_autocast, dtype=torch.float32)
        with amp_ctx:

            # print ("=*="*60)
            # print (f"PHRASES: {phrases}")
            # print (f"enc", enc)
            # print (f"labels: {labels}")

            # print("input_ids", enc["input_ids"].shape, "attention_mask", enc["attention_mask"].shape)

            outputs = model(**enc, labels=labels)
            loss_1 = outputs.loss 

            # print (f"STEP LOSS {loss_1.item()}")
            # print ("=*="*60)

            loss_dict = getattr(outputs, 'loss_dict', {})
            loss =  loss_1 / TrainingConfig.accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=15.0)

        if (step + 1) % TrainingConfig.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * TrainingConfig.accumulation_steps

        # estimate remaining time every 100 steps
        if global_step % 50 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed
            remaining_steps = total_steps - global_step
            est_remaining = remaining_steps / steps_per_sec
            est_total = total_steps / steps_per_sec

            print(f"epoch {epoch+1}/{TrainingConfig.epochs} step {step}/{len(train_dataloader)} "
                  f"Loss: {loss.item()*TrainingConfig.accumulation_steps:.4f} | "
                  f"Elapsed: {elapsed/60:.2f} min | "
                  f"ETA: {est_remaining/60:.2f} min | "
                  f"Total est: {est_total/60:.2f} min | "
                  f"Memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB , \\ {torch.mps.driver_allocated_memory() / 1e9:.2f} GB | "
                  )
            

    if (step + 1) % TrainingConfig.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()


    # del enc , labels, images, phrases
    torch.mps.empty_cache()
    import gc; gc.collect()
