import torch.nn as nn 
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

### Decoder Block ####


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

# Positional encoding helpers
def get_2d_sincos_pos_embed(h, w, dim):
    """Create standard 2D sine-cos positional embeddings (no learnable params).
    Args:
        h, w: spatial height & width
        dim: embedding dimension (must be even)
    Returns:
        Tensor shape (h*w, dim)
    """
    if dim % 4 != 0:
        # Fallback: pad to next multiple of 4 then slice
        full_dim = (dim // 4 + 1) * 4
    else:
        full_dim = dim
    pe = []
    # Coordinate grids normalized to [0,1]
    y = torch.linspace(0, 1, steps=h)
    x = torch.linspace(0, 1, steps=w)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    # Split dims equally for (x,y) sin/cos
    div_term = torch.exp(torch.arange(0, full_dim//4, 2, dtype=torch.float32) * (-math.log(10000.0) / (full_dim//4)))
    def encode(coord):
        # coord: (h,w)
        coord = coord.unsqueeze(-1)
        sinv = torch.sin(coord * div_term)
        cosv = torch.cos(coord * div_term)
        return torch.cat([sinv, cosv], dim=-1)
    y_enc = encode(yy)
    x_enc = encode(xx)
    # Concatenate along last dim -> (h,w, dim')
    pos = torch.cat([y_enc, x_enc], dim=-1)
    pos = pos.view(h*w, -1)
    if pos.shape[-1] > dim:
        pos = pos[:, :dim]
    elif pos.shape[-1] < dim:
        # Pad remaining
        pad = dim - pos.shape[-1]
        pos = torch.cat([pos, torch.zeros(h*w, pad)], dim=-1)
    return pos



class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        from transformers.models.detr.modeling_detr import DetrHungarianMatcher as HFMatcher
        self.hf = HFMatcher(class_cost=cost_class, bbox_cost=cost_bbox, giou_cost=cost_giou)
       
    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.hf({'pred_logits': outputs['class_logits'], 'pred_boxes': outputs['bboxes']}, targets)

class DeformableCrossAttention(nn.Module):
    """Simplified deformable cross-attention: samples sparse points around learned offsets.
    Not full MS-Deformable, but approximates focusing mechanism.
    Args:
        embed_dim: hidden size
        num_heads: attention heads
        num_points: sampled key points per head
    """
    def __init__(self, embed_dim, num_heads=8, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.num_points = num_points
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Offsets per head and point (relative index positions)
        self.offset_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_heads * num_points)
        )
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value):
        B, Q, D = query.shape
        _, K, _ = key.shape
        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(B, K, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(B, K, self.num_heads, self.head_dim)

        # Compute offsets -> indices
        offsets = self.offset_mlp(query).view(B, Q, self.num_heads, self.num_points)
        # Base index grid (uniform spanning K)
        base = torch.linspace(0, K - 1, steps=self.num_points, device=query.device)
        base = base.view(1, 1, 1, self.num_points).expand(B, Q, self.num_heads, self.num_points)
        # Apply offsets (small)
        offsets = torch.tanh(offsets) * (K / 32.0)
        idx = base + offsets
        idx = idx.clamp(0, K - 1)
        # Gather sampled keys/values
        idx_long = idx.round().long()  # (B,Q,H,P)
        k_sample = k.gather(1, idx_long.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim))  # (B,Q,H,P,head_dim)
        v_sample = v.gather(1, idx_long.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim))
        # Compute attention weights between q and sampled k
        q_exp = q.unsqueeze(3)  # (B,Q,H,1,head_dim)
        attn_logits = (q_exp * k_sample).sum(-1) * self.scale  # (B,Q,H,P)
        attn_weights = attn_logits.softmax(-1)
        # Weighted sum
        out = (attn_weights.unsqueeze(-1) * v_sample).sum(3)  # (B,Q,H,head_dim)
        out = out.view(B, Q, D)
        return self.out_proj(out), attn_weights.detach()
    

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = ['labels', 'boxes', 'giou']

    def loss_labels(self, outputs, targets, indices):
        """Simplified classification: only matched queries contribute; unmatched ignored."""
        src_logits = outputs['class_logits']
        matched_logits = []
        matched_targets = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if tgt_idx.numel() == 0:
                continue
            matched_logits.append(src_logits[b, src_idx])
            matched_targets.append(targets[b]['labels'][tgt_idx])

        if not matched_logits:
            return {'loss_ce': torch.zeros((), device=src_logits.device)}
        
        matched_logits = torch.cat(matched_logits, dim=0)
        matched_targets = torch.cat(matched_targets, dim=0)

        loss_ce = F.cross_entropy(matched_logits, matched_targets)

        return {'loss_ce': loss_ce}

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
        
        # Configurable depth
        self.num_blocks = 4  # increase from 2 to 4 blocks (each: self-attn + cross-attn)

        # 2. Cross-attention layers (Q-Former style) / optionally deformable
        self.use_deformable = True  # flag to enable deformable cross-attn
        if self.use_deformable:
            self.cross_attention_layers = nn.ModuleList([
                DeformableCrossAttention(embed_size, num_heads=num_heads, num_points=4)
                for _ in range(self.num_blocks)
            ])
        else:
            self.cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True, dropout=0.1)
                for _ in range(self.num_blocks)
            ])

        # 3. Self-attention layers for query refinement
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True, dropout=0.1)
            for _ in range(self.num_blocks)
        ])

        # 4. FFN layers after each attention (2 per block: after self-attn, after cross-attn)
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, 4 * embed_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(4 * embed_size, embed_size),
                nn.Dropout(0.1)
            )
            for _ in range(self.num_blocks * 2)
        ])

        # 5. Layer norms (4 per block: pre self, pre ffn1, pre cross, pre ffn2)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_size, eps=1e-6)
            for _ in range(self.num_blocks * 4)
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
        self.num_classes = 91  # raw COCO ids (1-90) plus slot for 0 if needed

        self.localization_ref_norm1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.localization_ref_attn = nn.MultiheadAttention(embed_size, 8, batch_first=True)
        self.localization_ref_norm2 = nn.LayerNorm(embed_size, eps=1e-6)
        self.localization_ref_ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU()
        )

        self.localization_head = nn.ModuleDict({
            "shared": nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.LayerNorm(embed_size, eps=1e-6),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            "bbox_head": nn.Sequential(
                nn.Linear(embed_size, embed_size // 2),
                nn.LayerNorm(embed_size // 2, eps=1e-6),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_size // 2, embed_size // 4),
                nn.GELU(),
                nn.Linear(embed_size // 4, 4)
            ),
            "bbox_refine_head": nn.Sequential(
                nn.Linear(embed_size, embed_size // 2),
                nn.GELU(),
                nn.Linear(embed_size // 2, 4)
            ),
            "objectness_head": nn.Sequential(
                nn.Linear(embed_size, embed_size // 2),
                nn.LayerNorm(embed_size // 2, eps=1e-6),
                nn.GELU(),
                nn.Linear(embed_size // 2, 1)
            ),
            "iou_head": nn.Sequential(
                nn.Linear(embed_size, embed_size // 2),
                nn.GELU(),
                nn.Linear(embed_size // 2, 1)
            ),
            "class_head": nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.LayerNorm(embed_size, eps=1e-6),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_size, self.num_classes)
            )
        })

        # Auxiliary prediction head (deep supervision) after first cross-attn block
        self.use_aux = True
        self.aux_head = nn.ModuleDict({
            "bbox_head": nn.Sequential(
                nn.Linear(embed_size, embed_size // 2),
                nn.GELU(),
                nn.Linear(embed_size // 2, 4)
            ),
            "class_head": nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.GELU(),
                nn.Linear(embed_size, self.num_classes)
            )
        })

        self.matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        self.criterion = SetCriterion(num_classes=self.num_classes, matcher=self.matcher,
                                weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2})

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        # Multiscale configuration
        self.use_multiscale = True  # flag to enable pyramid features
        self.pyramid_levels = 3     # number of additional pooled scales (besides base)
        if self.use_multiscale:
            self.scale_embed = nn.Embedding(self.pyramid_levels + 1, embed_size)
            self.fpn_reduce = nn.Linear(embed_size, embed_size)  # optional projection per scale

   

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
        cpu_device = torch.device("cpu")
        
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

    def _build_multiscale_features(self, img_features):
        """Construct pyramid features by average pooling progressively.
        Returns concatenated features with scale embeddings added.
        img_features: (B, N, D) flattened spatial tokens.
        Assumes original layout roughly square; reshapes for pooling.
        """
        B, N, D = img_features.shape
        h = int(math.sqrt(N)); w = h
        if h * w != N:
            # Fallback: treat sequence as (N,1)
            h, w = N, 1
        feats = img_features.view(B, h, w, D).permute(0, 3, 1, 2)  # (B,D,H,W)
        all_scales = []
        for lvl in range(self.pyramid_levels + 1):
            if lvl == 0:
                f = feats
            else:
                # progressive pooling by 2 each level
                f = F.avg_pool2d(feats, kernel_size=2**lvl, stride=2**lvl)
            b, d, hh, ww = f.shape
            flat = f.permute(0, 2, 3, 1).reshape(B, hh*ww, D)
            scale_emb = self.scale_embed.weight[lvl].unsqueeze(0).unsqueeze(1)  # (1,1,D)
            flat = flat + scale_emb
            # optional projection
            flat = self.fpn_reduce(flat)
            all_scales.append(flat)
        return torch.cat(all_scales, dim=1)  # (B, sum_tokens, D)
        
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
        """Clean forward with DN queries, deformable attention, attention pooling, and losses."""
        tok_embeds = self.gpt_decoder.get_input_embeddings()(captions_tensor)
        img_features = img_features.to(tok_embeds.device)
        B, T, D = tok_embeds.shape
        N = img_features.shape[1]

        # Positional embeddings
        if hasattr(self, 'pos_cache') and self.pos_cache.get(N) is not None:
            pos_embed = self.pos_cache[N].to(img_features.device)
        else:
            h = int(math.sqrt(N)); w = h
            if h*w != N: h, w = N, 1
            pos_embed = get_2d_sincos_pos_embed(h, w, D).to(img_features.device)
            if not hasattr(self, 'pos_cache'): self.pos_cache = {}
            self.pos_cache[N] = pos_embed.detach()
        img_features = img_features + pos_embed.unsqueeze(0)

        # Base queries and denoising queries
        base_query_tokens = self.query_tokens.expand(B, -1, -1)
        dn_query_tokens = None; dn_class_targets = None; dn_bbox_targets = None; dn_token_features = None
        if mode == 'train' and bbox_targets is not None and class_targets is not None:
            max_dn = 10
            dn_tokens=[]; dn_cls=[]; dn_boxes=[]
            for b in range(B):
                vm = class_targets[b] != -1
                lbl = class_targets[b][vm]; box = bbox_targets[b][vm]
                if box.numel()==0: continue
                take = min(lbl.size(0), max_dn)
                lbl = lbl[:take]; box = box[:take]
                noise = torch.randn_like(box)*0.05
                box_noisy = (box + noise).clamp(0,1)
                onehot = F.one_hot(lbl, num_classes=self.num_classes).float()
                concat = torch.cat([box_noisy, onehot], dim=-1)
                if not hasattr(self,'dn_proj'): self.dn_proj = nn.Linear(4 + self.num_classes, self.embed_size)
                dn_tok = self.dn_proj(concat)
                dn_tokens.append(dn_tok.unsqueeze(0)); dn_cls.append(lbl); dn_boxes.append(box)
            if dn_tokens:
                dn_query_tokens = torch.zeros(B, max_dn, self.embed_size, device=img_features.device)
                dn_class_targets = torch.full((B,max_dn), -1, dtype=torch.long, device=img_features.device)
                dn_bbox_targets = torch.zeros(B,max_dn,4, device=img_features.device)
                for b,dn_tok in enumerate(dn_tokens):
                    t = dn_tok.size(1)
                    dn_query_tokens[b,:t] = dn_tok.squeeze(0)
                    dn_class_targets[b,:t] = dn_cls[b]
                    dn_bbox_targets[b,:t] = dn_boxes[b]
                    
        query_tokens = torch.cat([dn_query_tokens, base_query_tokens], dim=1) if dn_query_tokens is not None else base_query_tokens
        img_ctx = self.img_context.expand(B,1,-1)
        query_tokens = torch.cat([img_ctx, query_tokens], dim=1)

        # Project image features (single or multiscale)
        if self.use_multiscale:
            multi_feats = self._build_multiscale_features(img_features)
            img_keys = self.key_proj(multi_feats)
            img_values = self.value_proj(multi_feats)
            key_token_count = img_keys.size(1)
        else:
            img_keys = self.key_proj(img_features)
            img_values = self.value_proj(img_features)
            key_token_count = img_keys.size(1)

        aux_bbox=None; aux_class_logits=None
        for b_i in range(self.num_blocks):
            ln = b_i*4; ffn = b_i*2
            nq = self.layer_norms[ln](query_tokens)
            self_out,_ = self.self_attention_layers[b_i](nq,nq,nq)
            query_tokens = query_tokens + self_out
            nq = self.layer_norms[ln+1](query_tokens)
            query_tokens = query_tokens + self.ffn_layers[ffn](nq)
            nq = self.layer_norms[ln+2](query_tokens)
            if self.use_deformable:
                cross_out, attn_w = self.cross_attention_layers[b_i](nq, img_keys, img_values)
            else:
                cross_out, attn_w = self.cross_attention_layers[b_i](query=nq, key=img_keys, value=img_values, need_weights=True, average_attn_weights=False)
            if b_i == 0: self.last_cross_attn_weights_1 = attn_w.detach().cpu()
            if b_i == 1: self.last_cross_attn_weights_2 = attn_w.detach().cpu()
            self.last_cross_attn_weights = attn_w.detach().cpu()
            query_tokens = query_tokens + cross_out
            if b_i == 0 and self.use_aux:
                aux_tokens = query_tokens[:,1:,:]
                aux_bbox = torch.sigmoid(self.aux_head['bbox_head'](aux_tokens))
                aux_class_logits = self.aux_head['class_head'](aux_tokens)
            nq = self.layer_norms[ln+3](query_tokens)
            query_tokens = query_tokens + self.ffn_layers[ffn+1](nq)
        visual_tokens = query_tokens

        global_img_context = visual_tokens[:,0:1,:]
        if dn_query_tokens is not None:
            dn_count = dn_query_tokens.size(1)
            img_token_features = visual_tokens[:,1+dn_count:,:]
            dn_token_features = visual_tokens[:,1:1+dn_count,:]
        else:
            img_token_features = visual_tokens[:,1:,:]
            dn_token_features = None
        if self.debug_similarity:
            self.compute_token_similarities(global_img_context, img_token_features, img_features)

        # Localization and refinement
        shared_features = self.localization_head['shared'](img_token_features)
        rf = self.localization_ref_norm1(shared_features)
        refined,_ = self.localization_ref_attn(rf, rf, rf)
        refine_x = shared_features + refined
        refine_x = refine_x + self.localization_ref_ffn(self.localization_ref_norm2(refine_x))
        shared_features = shared_features + refine_x
        bbox_raw = self.localization_head['bbox_head'](shared_features)
        bbox_preds = torch.sigmoid(bbox_raw)
        bbox_refine_delta = torch.tanh(self.localization_head['bbox_refine_head'](shared_features))*0.1
        bbox_refined = bbox_preds + bbox_refine_delta
        objectness_logits = self.localization_head['objectness_head'](shared_features)
        iou_logits = self.localization_head['iou_head'](shared_features)
        attn_source = getattr(self,'last_cross_attn_weights', None)
        fused_class_features = shared_features

        if attn_source is not None:
            attn_t = attn_source.to(shared_features.device)
            if attn_t.dim()==4 and attn_t.size(1)!=getattr(self,'num_heads',attn_t.size(1)):
                attn_t = attn_t.permute(0,2,1,3)
            attn_mean = attn_t.mean(1)
            attn_queries = attn_mean[:,1:,:]
            key_len = attn_queries.size(-1)
            # Choose corresponding base feats slice (multiscale may increase key length)
            if self.use_multiscale:
                # Recompute multi_feats (already computed as multi_feats above)
                base_feats = multi_feats[:, :key_len, :]
            else:
                base_feats = img_features[:, :key_len, :]
            pooled = torch.matmul(attn_queries, base_feats)
            fused_class_features = fused_class_features + pooled
        class_logits = self.localization_head['class_head'](fused_class_features)
        objectness_pred = torch.sigmoid(objectness_logits).squeeze(-1)
        iou_pred = torch.sigmoid(iou_logits).squeeze(-1)
        class_pred = class_logits.softmax(-1).argmax(-1)
        logits = torch.tensor([[1]], device=img_features.device)
        if mode != 'train':
            return logits, bbox_preds, objectness_pred, class_pred, [None]*6

        # Criterion targets
        outputs_for_criterion = {'bboxes': bbox_refined.detach(), 'class_logits': class_logits}
        targets_for_criterion = []
        for b in range(B):
            vm = class_targets[b] != -1
            lbl = class_targets[b][vm]; box = bbox_targets[b][vm]
            if box.numel()>0:
                wh = box[:,2:4]; ndm = (wh[:,0]>0) & (wh[:,1]>0)
                lbl = lbl[ndm]; box = box[ndm]
            targets_for_criterion.append({'labels': lbl, 'boxes': box})
        criterion_losses = self.criterion(outputs_for_criterion, targets_for_criterion)
        if self.use_aux and aux_bbox is not None:
            aux_outputs = {'bboxes': aux_bbox, 'class_logits': aux_class_logits}
            aux_losses = self.criterion(aux_outputs, targets_for_criterion)

        # Objectness focal loss
        with torch.no_grad():
            pos_mask = objectness_targets.squeeze(-1)==1
            neg_mask_all = objectness_targets.squeeze(-1)==0
            sampled_neg_mask = torch.zeros_like(neg_mask_all)
            for b in range(B):
                pc = pos_mask[b].sum().item(); neg_idx = torch.nonzero(neg_mask_all[b]).view(-1)
                if pc>0 and neg_idx.numel()>0:
                    k = min(pc, neg_idx.numel())
                    chosen = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:k]]
                    sampled_neg_mask[b, chosen] = True
            bce_mask = pos_mask | sampled_neg_mask
        masked_logits = objectness_logits.squeeze(-1)[bce_mask]
        masked_targets = objectness_targets.squeeze(-1)[bce_mask]
        def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
            if logits.numel()==0: return torch.zeros((), device=logits.device)
            prob = torch.sigmoid(logits); pt = prob*targets + (1-prob)*(1-targets)
            w = alpha*targets + (1-alpha)*(1-targets)
            loss_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            return (w * (1-pt).pow(gamma) * loss_bce).mean()
        objectness_loss = focal_loss(masked_logits, masked_targets)

        # IoU supervision
        matched_idx = self.matcher(outputs_for_criterion, targets_for_criterion)
        iou_supervision = torch.zeros((), device=img_features.device)
        if matched_idx:
            batch_ids=[]; pred_ids=[]; tgt_ious=[]
            for b_i,(src_i,tgt_i) in enumerate(matched_idx):
                if src_i.numel()==0: continue
                pred_m = bbox_refined[b_i, src_i]; tgt_m = bbox_targets[b_i][class_targets[b_i]!=-1][tgt_i]
                giou_mat = generalized_box_iou(box_cxcywh_to_xyxy(pred_m), box_cxcywh_to_xyxy(tgt_m))
                pair = torch.diag(giou_mat).clamp(0,1)
                batch_ids.append(torch.full_like(src_i, b_i)); pred_ids.append(src_i); tgt_ious.append(pair)
            if tgt_ious:
                batch_ids=torch.cat(batch_ids); pred_ids=torch.cat(pred_ids); tgt_ious=torch.cat(tgt_ious)
                pred_iou_vals = iou_pred[batch_ids, pred_ids]
                iou_supervision = F.smooth_l1_loss(pred_iou_vals, tgt_ious, reduction='mean')

        # Denoising direct losses
        dn_loss_ce = torch.zeros((), device=img_features.device); dn_loss_bbox = torch.zeros((), device=img_features.device); dn_loss_giou = torch.zeros((), device=img_features.device)
        if dn_token_features is not None and dn_class_targets is not None:
            dn_shared = self.localization_head['shared'](dn_token_features)
            dn_bbox = torch.sigmoid(self.localization_head['bbox_head'](dn_shared))
            dn_class_logits = self.localization_head['class_head'](dn_shared)
            dn_valid = dn_class_targets != -1
            if dn_valid.any():
                dn_loss_ce = F.cross_entropy(dn_class_logits[dn_valid], dn_class_targets[dn_valid])
                dn_loss_bbox = F.l1_loss(dn_bbox[dn_valid], dn_bbox_targets[dn_valid], reduction='mean')
                giou_dn = generalized_box_iou(box_cxcywh_to_xyxy(dn_bbox[dn_valid]), box_cxcywh_to_xyxy(dn_bbox_targets[dn_valid]))
                dn_loss_giou = 1 - torch.diag(giou_dn).mean()

        loss_bbox = criterion_losses['loss_bbox']; loss_giou = criterion_losses['loss_giou']; loss_ce = criterion_losses['loss_ce']
        lm_loss = torch.zeros((), device=img_features.device)
        total_loss = loss_bbox + loss_giou + loss_ce + 0.1*objectness_loss + 0.2*iou_supervision + 0.3*(dn_loss_ce+dn_loss_bbox+dn_loss_giou) + lm_loss
        loss_list = [total_loss, lm_loss, loss_bbox, loss_giou, loss_ce, objectness_loss, iou_supervision, dn_loss_ce, dn_loss_bbox, dn_loss_giou]
        if self.use_aux and aux_bbox is not None:
            total_loss = total_loss + 0.5*(aux_losses['loss_bbox'] + aux_losses['loss_giou'] + aux_losses['loss_ce'])
            loss_list.insert(6, aux_losses['loss_bbox']); loss_list.insert(7, aux_losses['loss_giou']); loss_list.insert(8, aux_losses['loss_ce']); loss_list[0] = total_loss
        return logits, bbox_preds, objectness_pred, class_pred, loss_list


        ###TO BE ON SAFE SIDE
        attention_mask = (captions_tensor != self.pad_token_id).long()

        # GPT self-attn runs first (inside decoder)
        #text_hidden_states = self.gpt_decoder.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state


        # Explicit cross-attention: text (query) attends to visual tokens (key/value)
        #cross_query = self.decoder_cross_attn_norm(text_hidden_states)

        # Create key padding mask for image (usually none)
        image_pad_mask = torch.zeros(B, visual_tokens.size(1), dtype=torch.bool, device=visual_tokens.device)


        # cross_out, cross_weights = self.decoder_cross_attn(
        #     query=cross_query,
        #     key=visual_tokens,
        #     value=visual_tokens,
        #     key_padding_mask=image_pad_mask,
        #     need_weights=True,
        #     average_attn_weights=False
        # )
        #self.decoder_last_cross_attn_weights = cross_weights.detach().cpu()

        #self.all_cross_attn.append(cross_weights.detach().cpu())

        # print (f"cross_out shape {cross_out.shape}")
        # print (f"cross_weights shape {cross_weights.shape}")


        # Residual + FFN
        # cross_out = text_hidden_states + cross_out
        #cross_out = text_hidden_states + self.cross_scale * cross_out




        #cross_out = cross_out + self.decoder_cross_attn_ffn(self.decoder_cross_attn_norm2(cross_out))
        #cross_out = self.dropout(cross_out)

        # ----- 6️  Prediction head -----
        #logits = self.gpt_decoder.lm_head(cross_out)
        # Localization head predictions

        # First pass through shared embedding layer
        shared_features = self.localization_head["shared"](img_token_features)

        refined_features = self.localization_ref_norm1(shared_features)
        refined, _ = self.localization_ref_attn(refined_features, refined_features, refined_features)
        refine_x = shared_features + refined
        refine_x = refine_x + self.localization_ref_ffn(self.localization_ref_norm2(refine_x))

        
        shared_features = shared_features + refine_x

    # Then through specialized prediction heads
    bbox_raw = self.localization_head["bbox_head"](shared_features)
    bbox_preds = torch.sigmoid(bbox_raw)
    bbox_refine_delta = torch.tanh(self.localization_head["bbox_refine_head"](shared_features)) * 0.1
    bbox_refined = bbox_preds + bbox_refine_delta  # local refinement
    objectness_logits = self.localization_head["objectness_head"](shared_features)
    iou_logits = self.localization_head["iou_head"](shared_features)
    
    # Attention-weight pooled features for class head
    # Use attention weights from last cross-attn block (stored in self.last_cross_attn_weights_2 if exists, else last_cross_attn_weights_1)
    attn_source = getattr(self, 'last_cross_attn_weights_2', None)
    if attn_source is None:
        attn_source = getattr(self, 'last_cross_attn_weights_1', None)
    if attn_source is not None:
        # attn_source shape: (B, num_heads, query_len, key_len)
        # We exclude global token for class predictions (query_tokens without first)
        attn_weights_final = attn_source  # cpu tensor
        attn_weights_final = attn_weights_final.to(shared_features.device)
        # Aggregate heads
        attn_agg = attn_weights_final.mean(1)  # (B, Q, K)
        # Remove global query (assumed first)
        attn_queries = attn_agg[:, 1:, :]  # (B, num_img_tokens, K)
        # Project original img_features (B,N,D) -> we already added positional embed; pool over keys (N)
        # If key_len differs from N (e.g., due to padding), clamp
        key_len = attn_queries.size(-1)
        base_img_feats = img_features[:, :key_len, :]
        pooled = torch.matmul(attn_queries, base_img_feats)  # (B, num_img_tokens, D)
        # Fuse pooled with shared_features
        fused_class_features = shared_features + pooled
    else:
        fused_class_features = shared_features
    class_logits = self.localization_head["class_head"](fused_class_features)

    objectness_pred = torch.sigmoid(objectness_logits).squeeze(-1)
    iou_pred = torch.sigmoid(iou_logits).squeeze(-1)
    class_pred = class_logits.softmax(-1).argmax(-1)

    # ----- 7  Loss -----
    loss = None

    logits = torch.tensor([[1]])
    if mode == "train":
            #loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            #shift_logits = logits[:, :-1, :].contiguous()
            #shift_labels = captions_tensor[:, 1:].contiguous()
            #lm_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

            # 2. Hungarian + SetCriterion for boxes + class
            outputs_for_criterion = {
                "bboxes": bbox_refined.detach(),  # use refined for matching, detach to keep gradients stable
                "class_logits": class_logits
            }
            targets_for_criterion = []
            for b in range(B):
                # filter padded labels
                valid_mask = class_targets[b] != -1
                filtered_labels = class_targets[b][valid_mask]
                filtered_boxes = bbox_targets[b][valid_mask]
                if filtered_boxes.numel() > 0:
                    # boxes are cx,cy,w,h normalized; zero-area if w==0 or h==0
                    box_wh = filtered_boxes[:, 2:4]
                    nondeg_mask = (box_wh[:, 0] > 0) & (box_wh[:, 1] > 0)
                    filtered_labels = filtered_labels[nondeg_mask]
                    filtered_boxes = filtered_boxes[nondeg_mask]
                targets_for_criterion.append({
                    "labels": filtered_labels,
                    "boxes": filtered_boxes
                })

            criterion_losses = self.criterion(outputs_for_criterion, targets_for_criterion)
            # Denoising losses (direct supervision, skip matcher)
            dn_loss_ce = torch.zeros((), device=tok_embeds.device)
            dn_loss_bbox = torch.zeros((), device=tok_embeds.device)
            dn_loss_giou = torch.zeros((), device=tok_embeds.device)
            if dn_token_features is not None:
                # Predict for dn tokens using same heads
                dn_shared = self.localization_head["shared"](dn_token_features)
                dn_bbox = torch.sigmoid(self.localization_head["bbox_head"](dn_shared))
                dn_class_logits = self.localization_head["class_head"](dn_shared)
                # Valid mask
                dn_valid = dn_class_targets != -1
                if dn_valid.any():
                    dn_flat_logits = dn_class_logits[dn_valid]
                    dn_flat_targets = dn_class_targets[dn_valid]
                    dn_loss_ce = F.cross_entropy(dn_flat_logits, dn_flat_targets)
                    dn_flat_bbox = dn_bbox[dn_valid]
                    dn_flat_tgt_bbox = dn_bbox_targets[dn_valid]
                    dn_loss_bbox = F.l1_loss(dn_flat_bbox, dn_flat_tgt_bbox, reduction='mean')
                    dn_flat_bbox_xyxy = box_cxcywh_to_xyxy(dn_flat_bbox)
                    dn_flat_tgt_xyxy = box_cxcywh_to_xyxy(dn_flat_tgt_bbox)
                    giou_mat = generalized_box_iou(dn_flat_bbox_xyxy, dn_flat_tgt_xyxy)
                    dn_loss_giou = 1 - torch.diag(giou_mat).mean()
            if self.use_aux:
                aux_outputs = {"bboxes": aux_bbox, "class_logits": aux_class_logits}
                aux_losses = self.criterion(aux_outputs, targets_for_criterion)

            # Optional: Add objectness loss
            # Masked objectness loss: compute BCE only on real objects and a sample of negatives

            with torch.no_grad():
                pos_mask = objectness_targets.squeeze(-1) == 1
                neg_mask_all = objectness_targets.squeeze(-1) == 0
                # sample at most same number of negatives as positives for balance
                sampled_neg_mask = torch.zeros_like(neg_mask_all)
                for b in range(B):
                    pos_count = pos_mask[b].sum().item()
                    neg_indices = torch.nonzero(neg_mask_all[b]).view(-1)
                    if pos_count > 0 and neg_indices.numel() > 0:
                        k = min(pos_count, neg_indices.numel())
                        sampled = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:k]]
                        sampled_neg_mask[b, sampled] = True
                bce_mask = pos_mask | sampled_neg_mask
            masked_logits = objectness_logits.squeeze(-1)[bce_mask]
            masked_targets = objectness_targets.squeeze(-1)[bce_mask]

            # Focal loss for objectness
            def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
                if logits.numel() == 0:
                    return torch.zeros((), device=logits.device)
                prob = torch.sigmoid(logits)
                pt = prob * targets + (1 - prob) * (1 - targets)
                w = alpha * targets + (1 - alpha) * (1 - targets)
                loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                loss = w * (1 - pt).pow(gamma) * loss
                return loss.mean()

            objectness_loss = focal_loss(masked_logits, masked_targets)

            loss_bbox = criterion_losses["loss_bbox"]
            loss_giou = criterion_losses["loss_giou"]

            if self.use_aux:
                aux_loss_bbox = aux_losses["loss_bbox"]
                aux_loss_giou = aux_losses["loss_giou"]
                aux_loss_ce = aux_losses["loss_ce"]

            lm_loss = torch.zeros(B, device=tok_embeds.device, requires_grad=True).mean()


            # Total loss
            # IoU supervision: match predicted IoU for the matched queries (reuse indices from criterion)
            matched_idx = self.criterion.matcher(outputs_for_criterion, targets_for_criterion)
            iou_supervision = torch.zeros((), device=tok_embeds.device)
            if matched_idx:
                batch_ids = []
                pred_ids = []
                target_ious = []
                for b_i, (src_i, tgt_i) in enumerate(matched_idx):
                    if src_i.numel() == 0:
                        continue
                    # predicted refined boxes for matched queries
                    matched_pred = bbox_refined[b_i, src_i]
                    matched_tgt = bbox_targets[b_i][class_targets[b_i] != -1][tgt_i]
                    pred_xyxy = box_cxcywh_to_xyxy(matched_pred)
                    tgt_xyxy = box_cxcywh_to_xyxy(matched_tgt)
                    giou_mat = generalized_box_iou(pred_xyxy, tgt_xyxy)
                    # diag as pairwise matches
                    pair_giou = torch.diag(giou_mat)
                    batch_ids.append(torch.full_like(src_i, b_i))
                    pred_ids.append(src_i)
                    target_ious.append(pair_giou.clamp(0,1))
                if target_ious:
                    batch_ids = torch.cat(batch_ids)
                    pred_ids = torch.cat(pred_ids)
                    target_ious = torch.cat(target_ious)
                    pred_iou_vals = iou_pred[batch_ids, pred_ids]
                    iou_supervision = F.smooth_l1_loss(pred_iou_vals, target_ious, reduction='mean')
            total_loss = loss_bbox + loss_giou + criterion_losses["loss_ce"] + 0.1 * objectness_loss + 0.2 * iou_supervision + lm_loss + 0.3 * (dn_loss_ce + dn_loss_bbox + dn_loss_giou)
            if self.use_aux:
                total_loss = total_loss + 0.5 * (aux_loss_bbox + aux_loss_giou + aux_loss_ce)

            if self.use_aux:
                loss_list = [total_loss, lm_loss, loss_bbox, loss_giou, criterion_losses['loss_ce'], objectness_loss,
                             aux_loss_bbox, aux_loss_giou, aux_loss_ce, iou_supervision, dn_loss_ce, dn_loss_bbox, dn_loss_giou]
            else:
                loss_list = [total_loss, lm_loss, loss_bbox, loss_giou, criterion_losses['loss_ce'], objectness_loss, iou_supervision, dn_loss_ce, dn_loss_bbox, dn_loss_giou]

            return logits, bbox_preds, objectness_pred, class_pred, loss_list

    else:  # ----- mode == "inference" -----
            # Mask padding tokens to avoid attention to them
            # if attention_mask is None:
            #     attention_mask = (captions_tensor != self.pad_token_id).long()

            # probs = F.softmax(logits, dim=-1)
            loss_list = [None, None, None , None , None, None ]
            return logits, bbox_preds, objectness_pred, class_pred, loss_list


      
       