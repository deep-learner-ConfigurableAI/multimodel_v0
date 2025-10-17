import torch.nn as nn 
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

### Decoder Block ####

device = torch.device("mps")

import torch
from scipy.optimize import linear_sum_assignment

from torch import nn

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)


def generalized_box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)

    lt_enclose = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enclose = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enclose = (rb_enclose - lt_enclose).clamp(min=0)
    area_enclose = wh_enclose[:, :, 0] * wh_enclose[:, :, 1]

    giou = iou - (area_enclose - union) / (area_enclose + 1e-7)
    return giou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: dict of tensors
          - "bboxes": [B, num_queries, 4]
          - "class_logits": [B, num_queries, num_classes]
        targets: list of dicts
          Each dict has:
            - "labels": Tensor[num_gt]
            - "boxes": Tensor[num_gt, 4]
        """
        bs, num_queries = outputs["class_logits"].shape[:2]
        out_prob = outputs["class_logits"].softmax(-1)  # (B, num_queries, num_classes)
        out_bbox = outputs["bboxes"]                   # (B, num_queries, 4)

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            # Compute classification cost
            cost_class = -out_prob[b][:, tgt_ids]

            # L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)

            # GIoU cost
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[b]), 
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = (self.cost_class * cost_class
               + self.cost_bbox * cost_bbox
               + self.cost_giou * cost_giou)

            C = C.cpu()
            i, j = linear_sum_assignment(C)
            indices.append((torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64)))

        return indices
    
    
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = ['labels', 'boxes', 'giou']

    def loss_labels(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['class_logits'][idx]

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:1], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes = target_classes.to(src_logits.device)
        loss_ce = F.cross_entropy(src_logits, target_classes_o)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['bboxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / src_boxes.shape[0]
        return {"loss_bbox": loss_bbox}

    def loss_giou(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = box_cxcywh_to_xyxy(outputs['bboxes'][idx])
        target_boxes = box_cxcywh_to_xyxy(torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0))
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes)).mean()
        return {"loss_giou": loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return (batch_idx, src_idx)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f"loss_{loss}")(outputs, targets, indices))
        return losses



class ResnetGPT2Wrapper(nn.Module):
    def __init__(self, gpt_decoder, embed_size, vocab_size, pad_token_id, num_heads=8, num_img_tokens=64, debug_similarity=False):
        super().__init__()
        self.gpt_decoder = gpt_decoder
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_img_tokens = num_img_tokens
        self.pad_token_id = pad_token_id
        self.debug_similarity = debug_similarity  # Flag to activate similarity debugging
        
        # For similarity visualization
        self.similarity_logs = []
        self.all_cross_attn = []
        
        # Q-Former style components
        # 1. Learnable query embeddings
        self.query_tokens = nn.Parameter(torch.randn(1, num_img_tokens, embed_size) * 0.02)
        
        # 2. Cross-attention layers (Q-Former has multiple layers)
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True, dropout=0.1)
            for _ in range(2)  # Using 2 cross-attention layers
        ])
        
        # 3. Self-attention layers for query refinement
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True, dropout=0.1)
            for _ in range(2)  # Using 2 self-attention layers
        ])
        
        # 4. FFN layers after each attention
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, 4 * embed_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(4 * embed_size, embed_size),
                nn.Dropout(0.1)
            )
            for _ in range(4)  # One for each attention layer
        ])
        
        # 5. Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_size, eps=1e-6)
            for _ in range(8)  # 2 per attention+ffn block
        ])
        
        # Final projections
        self.key_proj = nn.Linear(embed_size, embed_size)
        self.value_proj = nn.Linear(embed_size, embed_size)
        self.query_proj = nn.Linear(embed_size, embed_size)
        
        # Global image context for conditioning
        self.img_context = nn.Parameter(torch.randn(1, 1, embed_size) * 0.02)
        
        # Output dropout
        self.dropout = nn.Dropout(0.1)

        #### DECODER CROSS ATTN WHICH BRIDGES TEXT AND QUERY TOKENS 

        self.decoder_cross_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.decoder_cross_attn_norm = nn.LayerNorm(embed_size)
        self.decoder_cross_attn_ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(0.1)
        )
        self.decoder_cross_attn_norm2 = nn.LayerNorm(embed_size)

        self.decoder_last_cross_attn_weights = None 

        self.to("mps")

        self.cross_scale = nn.Parameter(torch.ones(1))


        ####  localization head for bounding box prediction

        self.num_classes = 80  #COCO classes 

        


        self.localization_head = nn.ModuleDict({
            "bbox_head": nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, 4),  # [cx, cy, w, h] normalized (0–1)
            ),
            "objectness_head": nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, 1),  # sigmoid → probability
            ),
            "class_head": nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, self.num_classes)  # softmax over classes
            )
        })


        self.matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        self.criterion = SetCriterion(num_classes=80, matcher=self.matcher,
                                weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2})



        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"


    def perform_mha_on_cpu(self, queries, k, v, attention_layer=None): 
        """
        Fallback method for multi-head attention when MPS has issues
        Can be used with any provided attention layer
        
        Args:
            queries: Query tensor
            k: Key tensor
            v: Value tensor
            attention_layer: Specific attention layer to use (defaults to first cross-attention layer)
        """
        # Default to first cross-attention layer if none specified
        if attention_layer is None:
            attention_layer = self.cross_attention_layers[0]
        
        # Process on CPU for stability
        cpu_device = torch.device("mps")
        
        queries_cpu = queries.to(cpu_device)
        k_cpu = k.to(cpu_device)
        v_cpu = v.to(cpu_device)
        
        # Move attention layer to CPU
        attention_cpu = attention_layer.to(cpu_device)
        
        # Run attention
        output, _ = attention_cpu(queries_cpu, k_cpu, v_cpu)
        
        # Move attention layer back
        attention_layer.to("mps")
        
        return output.to("mps")
        
    def compute_token_similarities(self, global_context, query_tokens, img_features):

        """
        Compute cosine similarity between tokens to analyze what global token is learning
        
        Args:
            global_context: Global context token (B, 1, D)
            query_tokens: Visual query tokens (B, num_img_tokens, D)
            img_features: Original image features from encoder (B, N, D)
        """
        
        # Get first batch for analysis
        global_ctx = global_context[0, 0]  # (D,)
        query_toks = query_tokens[0]  # (num_img_tokens, D)
        img_feats = img_features[0]  # (N, D)
        
        # Normalize all vectors for cosine similarity
        global_ctx_norm = F.normalize(global_ctx, p=2, dim=0)
        query_toks_norm = F.normalize(query_toks, p=2, dim=1)
        img_feats_norm = F.normalize(img_feats, p=2, dim=1)
        
        # Compute similarities
        # 1. Global context to query tokens
        global_to_query_sim = torch.matmul(query_toks_norm, global_ctx_norm)
        
        # 2. Global context to image features
        global_to_img_sim = torch.matmul(img_feats_norm, global_ctx_norm)
        
        # 3. Average query token to image features similarity
        query_to_img_sim = torch.matmul(img_feats_norm, query_toks_norm.transpose(0, 1))
        avg_query_to_img_sim = query_to_img_sim.mean(dim=1)

        global_to_query_sim = global_to_query_sim.to(torch.float32)
        global_to_img_sim = global_to_img_sim.to(torch.float32)
        avg_query_to_img_sim = avg_query_to_img_sim.to(torch.float32)


        
        # Log similarities for later visualization
        self.similarity_logs.append({
            'global_to_query': global_to_query_sim.detach().cpu().numpy(),
            'global_to_img': global_to_img_sim.detach().cpu().numpy(),
            'avg_query_to_img': avg_query_to_img_sim.detach().cpu().numpy(),
            'step': len(self.similarity_logs) + 1
        })
        
        # Print some statistics for immediate feedback
        print(f"--- Token Similarity Analysis [Step {len(self.similarity_logs)}] ---")
        print(f"Global-Query: mean={global_to_query_sim.mean().item():.4f}, max={global_to_query_sim.max().item():.4f}")
        print(f"Global-Image: mean={global_to_img_sim.mean().item():.4f}, max={global_to_img_sim.max().item():.4f}")
        print(f"Query-Image: mean={avg_query_to_img_sim.mean().item():.4f}, max={avg_query_to_img_sim.max().item():.4f}")
        
    def visualize_similarities(self, save_path=None):
        """
        Visualize the token similarities over time
        
        Args:
            save_path: Path to save the visualization, if None will display only
        """
        if not self.similarity_logs:
            print("No similarity logs available. Set debug_similarity=True during initialization.")
            return
            
        # Extract data from logs
        steps = [log['step'] for log in self.similarity_logs]
        global_query_means = [log['global_to_query'].mean() for log in self.similarity_logs]
        global_img_means = [log['global_to_img'].mean() for log in self.similarity_logs]
        query_img_means = [log['avg_query_to_img'].mean() for log in self.similarity_logs]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, global_query_means, 'b-', label='Global-Query Similarity')
        plt.plot(steps, global_img_means, 'r-', label='Global-Image Similarity')
        plt.plot(steps, query_img_means, 'g-', label='Query-Image Similarity')
        
        plt.xlabel('Training Step')
        plt.ylabel('Average Cosine Similarity')
        plt.title('Token Representation Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
            
    def plot_attention_heatmap(self, step_idx=-1, save_path=None):
        """
        Plot heatmap of token similarities at a specific step
        
        Args:
            step_idx: Index of the step to visualize (-1 for most recent)
            save_path: Path to save the visualization, if None will display only
        """
        if not self.similarity_logs:
            print("No similarity logs available. Set debug_similarity=True during initialization.")
            return
            
        log = self.similarity_logs[step_idx]
        global_to_query = log['global_to_query']
        
        # Create heatmap
        plt.figure(figsize=(10, 4))
        plt.imshow(global_to_query.reshape(1, -1), cmap='viridis', aspect='auto')
        plt.colorbar(label='Cosine Similarity')
        plt.xlabel('Query Token Index')
        plt.title(f'Global-Query Token Similarity (Step {log["step"]})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Heatmap saved to {save_path}")
        else:
            plt.show()

    def forward(self, img_features, captions_tensor, attention_mask=None, bbox_targets=None, class_targets=None, objectness_targets=None, mode="train"):
        """
        Q-Former style forward pass for image-text integration
        
        Args:
            img_features: Tensor of shape (B, N, D) - image features from encoder
            captions_tensor: Tensor of shape (B, T) - caption token ids
            attention_mask: Tensor of shape (B, T) - attention mask for captions
            mode: 'train' or 'inference' - determines whether to compute loss
        """
        # Get token embeddings from the decoder model
        tok_embeds = self.gpt_decoder.get_input_embeddings()(captions_tensor)

        # print (f"img_features: {img_features.shape}, tok_embeds: {tok_embeds.shape}")


        # Ensure everything is on the same device
        img_features = img_features.to(tok_embeds.device)
        
        # Get batch size and dimensions
        B, T, D = tok_embeds.shape
        N = img_features.shape[1]
        
        # Expand query tokens to match batch size
        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, num_img_tokens, D)
        
        # Add global context token to the beginning of query tokens
        img_ctx = self.img_context.expand(B, 1, -1)  # (B, 1, D)
        query_tokens = torch.cat([img_ctx, query_tokens], dim=1)  # (B, 1+num_img_tokens, D)

        # print (f"query_tokens after adding global context: {query_tokens.shape}")

        
        # Q-Former style processing with alternating self-attention and cross-attention
        
        # Layer 1: Self-attention for query tokens
        norm_query = self.layer_norms[0](query_tokens)

        self_attn_output, _ = self.self_attention_layers[0](
            query=norm_query,
            key=norm_query,
            value=norm_query
        )

        query_tokens = query_tokens + self_attn_output
        
        # Layer 1: FFN after self-attention
        norm_query = self.layer_norms[1](query_tokens)
        ffn_output = self.ffn_layers[0](norm_query)
        query_tokens = query_tokens + ffn_output


        # print ("Cross Attention about to start")

        # print ("DTYPE AND DEVICE CHECK")
        # print (f"norm_query: {norm_query.dtype}, {norm_query.device}")
        # print (f"img_features: {img_features.dtype}, {img_features.device}")
        # print (f"query_tokens: {query_tokens.dtype}, {query_tokens.device}")
        # print (f"key_proj: {next(self.key_proj.parameters()).dtype}, {next(self.key_proj.parameters()).device}")
        # print (f"value_proj: {next(self.value_proj.parameters()).dtype}, {next(self.value_proj.parameters()).device}")
        # print (f"self.img_context: {self.img_context.dtype}, {self.img_context.device}")

    

        # Layer 1: Cross-attention with image features
        norm_query = self.layer_norms[2](query_tokens)

        img_keys = self.key_proj(img_features)

        img_values = self.value_proj(img_features)


        cross_attn_output, attn_weights_1 = self.cross_attention_layers[0](
            query=norm_query,
            key=img_keys,
            value=img_values,
            need_weights=True,
            average_attn_weights=False
        )

        self.last_cross_attn_weights_1 = attn_weights_1.detach().cpu()


        query_tokens = query_tokens + cross_attn_output
        
        # Layer 1: FFN after cross-attention
        norm_query = self.layer_norms[3](query_tokens)
        ffn_output = self.ffn_layers[1](norm_query)
        query_tokens = query_tokens + ffn_output
        


        # Layer 2: Self-attention for query tokens
        norm_query = self.layer_norms[4](query_tokens)
        self_attn_output, _ = self.self_attention_layers[1](
            query=norm_query,
            key=norm_query,
            value=norm_query
        )

        query_tokens = query_tokens + self_attn_output
        
        # Layer 2: FFN after self-attention
        norm_query = self.layer_norms[5](query_tokens)
        ffn_output = self.ffn_layers[2](norm_query)
        query_tokens = query_tokens + ffn_output
        
        # Layer 2: Cross-attention with image features
        norm_query = self.layer_norms[6](query_tokens)

        cross_attn_output, attn_weights_2 = self.cross_attention_layers[1](
            query=norm_query,
            key=img_keys,
            value=img_values,
            need_weights=True,
            average_attn_weights=False,
        )

        self.last_cross_attn_weights_2 = attn_weights_2.detach().cpu()

        query_tokens = query_tokens + cross_attn_output
        
        # Layer 2: Final FFN
        norm_query = self.layer_norms[7](query_tokens)
        ffn_output = self.ffn_layers[3](norm_query)
        visual_tokens = query_tokens + ffn_output
        
        # Extract global image context and query tokens
        global_img_context = visual_tokens[:, 0:1, :]
        img_token_features = visual_tokens[:, 1:, :]
        
        # Calculate cosine similarity if debug flag is enabled
        if self.debug_similarity:
            self.compute_token_similarities(global_img_context, img_token_features, img_features)
        

        ###NEED DEEP_RESEARCH ON THIS IF THIS MAKE SENSE 
        # Conditioning: Use global context to influence text embeddings
        #conditioned_tok_embeds = tok_embeds + global_img_context.expand(-1, T, -1) * 0.2

        # ----- Prepare decoder text embeddings -----
        #text_embeds = tok_embeds + 0.2 * global_img_context.expand(-1, T, -1)  # light conditioning only

        text_embeds = tok_embeds   # removing global context influence to see impact of just cross or query tokens



        # print (f"conditioned_tok_embeds: {conditioned_tok_embeds.shape}")


        
        # Concatenate visual tokens with conditioned text embeddings
        #fused = torch.cat([img_token_features, conditioned_tok_embeds], dim=1)
        # fused = torch.cat([visual_tokens, conditioned_tok_embeds], dim=1)


        # print (f"fused: {fused.shape}")
        
        # Final dropout for regularization
        #inputs_embeds = self.dropout(fused).contiguous()

        inputs_embeds = self.dropout(text_embeds).contiguous()


        ###TO BE ON SAFE SIDE
        attention_mask = (captions_tensor != self.pad_token_id).long()

        # GPT self-attn runs first (inside decoder)
        text_hidden_states = self.gpt_decoder.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state


        # Explicit cross-attention: text (query) attends to visual tokens (key/value)
        cross_query = self.decoder_cross_attn_norm(text_hidden_states)

        # Create key padding mask for image (usually none)
        image_pad_mask = torch.zeros(B, visual_tokens.size(1), dtype=torch.bool, device=visual_tokens.device)


        cross_out, cross_weights = self.decoder_cross_attn(
            query=cross_query,
            key=visual_tokens,
            value=visual_tokens,
            key_padding_mask=image_pad_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.decoder_last_cross_attn_weights = cross_weights.detach().cpu()

        self.all_cross_attn.append(cross_weights.detach().cpu())

        # print (f"cross_out shape {cross_out.shape}")
        # print (f"cross_weights shape {cross_weights.shape}")


        # Residual + FFN
        # cross_out = text_hidden_states + cross_out
        cross_out = text_hidden_states + self.cross_scale * cross_out




        cross_out = cross_out + self.decoder_cross_attn_ffn(self.decoder_cross_attn_norm2(cross_out))
        cross_out = self.dropout(cross_out)

        # ----- 6️  Prediction head -----
        logits = self.gpt_decoder.lm_head(cross_out)
        # Localization head predictions
        bbox_preds = torch.sigmoid(self.localization_head["bbox_head"](img_token_features))  # (B, num_tokens, 4)
        objectness_logits = self.localization_head["objectness_head"](img_token_features)    # (B, num_tokens, 1)
        class_logits = self.localization_head["class_head"](img_token_features)              # (B, num_tokens, num_classes)

        objectness_pred = torch.sigmoid(objectness_logits).squeeze(-1)  # (B, num_tokens)
        class_pred = class_logits.argmax(dim=-1)  # (B, num_tokens)

        # ----- 7  Loss -----

        loss = None
        if mode == "train":
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = captions_tensor[:, 1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

            # 2. Hungarian + SetCriterion for boxes + class
            outputs_for_criterion = {
                "bboxes": bbox_preds,
                "class_logits": class_logits
            }
            targets_for_criterion = []
            for b in range(B):
                targets_for_criterion.append({
                    "labels": class_targets[b],
                    "boxes": bbox_targets[b]
                })

            criterion_losses = self.criterion(outputs_for_criterion, targets_for_criterion)
            # Optional: Add objectness loss
            objectness_loss = F.binary_cross_entropy_with_logits(objectness_logits, objectness_targets)

            # Total loss
            total_loss = lm_loss + criterion_losses["loss_bbox"] + criterion_losses["loss_giou"] + criterion_losses["loss_ce"] + 0.5 * objectness_loss
            return logits, bbox_preds, objectness_pred, class_pred, total_loss

        else:  # ----- mode == "inference" -----
            # Mask padding tokens to avoid attention to them
            if attention_mask is None:
                attention_mask = (captions_tensor != self.pad_token_id).long()

            probs = F.softmax(logits, dim=-1)
            return logits, bbox_preds, objectness_pred, class_pred, None 


        # if attention_mask is not None:
        #     B, T = captions_tensor.size()
        #     # Total number of visual tokens (query tokens + global context)
        #     #img_tokens = self.num_img_tokens + 1  # +1 for global context
        #     img_tokens = self.num_img_tokens  # +1 for global context

        #     #seq_len = inputs_embeds.size(1)  # num_img_tokens + T (text)

        #     seq_len = inputs_embeds.size(1)  #T (text)


        #     print (f"seq_length :\t {seq_len}")

        #     # --- Step 1: Create bidirectional mask for visual tokens ---
        #     # Q-Former visual tokens use bidirectional attention among themselves
        #     img_mask = torch.ones(B, img_tokens, device=tok_embeds.device)
            
        #     # --- Step 2: Combine with text attention mask ---
        #     #full_mask = torch.cat([img_mask, attention_mask], dim=1)  # (B, img_tokens+T)

        #     full_mask = attention_mask  # (B, T)


        #     # print (f"full_mask {full_mask.shape}")
            
        #     # --- Step 3: Create Q-Former style attention pattern ---
        #     # Create bidirectional mask for visual tokens and causal mask for text tokens
        #     # Visual tokens see all visual tokens, text tokens see all visual tokens and previous text tokens
            
        #     # Initialize with zeros
        #     full_attention_mask = torch.zeros((seq_len, seq_len), device=tok_embeds.device)


        #     # print (f"full_attention_mask {full_attention_mask.shape}")
        #     # print (f"img_tokens {img_tokens}")

            
        #     # Visual-to-visual: bidirectional attention (all 1s in top-left)
        #     # full_attention_mask[:img_tokens, :img_tokens] = 1
            
        #     # Text-to-visual: full attention to all visual tokens (all 1s in bottom-left)
        #     # full_attention_mask[img_tokens:, :img_tokens] = 1
            
        #     # Text-to-text: causal attention (lower triangle in bottom-right)
        #     # This ensures autoregressive text generation
        #     text_causal_mask = torch.tril(torch.ones((T, T), device=tok_embeds.device))

        #     # print (f"SHAPE ATTENTION", "*"*10)

        #     # print (full_attention_mask[img_tokens:, img_tokens:].shape)



        #     full_attention_mask[img_tokens:, img_tokens:] = text_causal_mask
            
        #     # Expand to 4D for batch and attention heads
        #     full_attention_mask = full_attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            
        #     # Create padding mask from the original attention mask
        #     padding_mask = full_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq_len)
            
        #     # Combine attention pattern with padding mask
        #     attention_mask = full_attention_mask * padding_mask



        # # Prepare labels for training
        # labels = None 
        # if mode == 'train':
        #     # For text tokens, we use the original caption tokens
        #     labels = captions_tensor[:, :].contiguous()
            
        #     # For visual tokens, we use padding tokens (these will be ignored in loss)
        #     # Include all visual tokens (query tokens + global context)
        #     pad_for_img = torch.full((B, self.num_img_tokens + 1), 
        #                             self.pad_token_id, 
        #                             dtype=torch.long, 
        #                             device=tok_embeds.device)
            
        #     # Concatenate padding for visual tokens with text labels
        #     labels = torch.cat([pad_for_img, labels], dim=1)   # (B, 1+num_img_tokens+T)


        # if mode == "train":
        #     outputs = self.gpt_decoder(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         labels=labels
        #     )
        # else:  # inference
        #     outputs = self.gpt_decoder(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask
        #     )


        # outputs = self.gpt_decoder(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     img_feats=img_features, 
        #     labels=labels
        # )

        # return outputs.logits, outputs.loss 
        #return outputs["logits"], outputs["loss"]
