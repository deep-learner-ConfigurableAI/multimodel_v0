from transformers import AutoProcessor
from transformers.models.grounding_dino import GroundingDinoForObjectDetection
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel
from PIL import Image
import dotenv
from contextlib import contextmanager
from typing import ClassVar
from pydantic import BaseModel
from torch.utils.data import DataLoader

processor = AutoProcessor.from_pretrained("grounding_dino_tiny_local")
tokenizer = processor.tokenizer


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    # Stack image pixel values
    pixel_values = torch.stack([item["dino"]["pixel_values"] for item in batch])
    clip_pixel_values = torch.stack([item["clip"]["pixel_values"] for item in batch])

    # Detect detection-only multi-query mode by presence of query_names in targets
    detection_mode = any(
        "query_names" in t for item in batch for t in item["dino"]["targets"]
    )

    targets = []
    print (f"\n\t detection_mode ::: {detection_mode}")

    if detection_mode:
        # Flatten queries across batch; adjust class_labels indices with offsets
        all_query_input_ids = []
        all_query_attention = []
        query_batch_index = []  # maps flattened query row -> sample index
        offset = 0

        for b_idx, item in enumerate(batch):
            sample_input_ids = item["dino"]["input_ids"]  # [Q, L] or [L]
            sample_attn = item["dino"]["attention_mask"]

            # Ensure 2D shape for uniform processing
            if sample_input_ids.dim() == 1:
                sample_input_ids = sample_input_ids.unsqueeze(0)
                sample_attn = sample_attn.unsqueeze(0)

            Q = sample_input_ids.size(0)
            for q in range(Q):
                all_query_input_ids.append(sample_input_ids[q])
                all_query_attention.append(sample_attn[q])
                query_batch_index.append(b_idx)

            # Adjust class_labels in target dict(s) of this sample
            for t in item["dino"]["targets"]:
                if "class_labels" in t and t["class_labels"].numel() > 0:
                    t["class_labels"] = t["class_labels"] + offset
                targets.append(t)

            offset += Q

        # Pad variable-length query sequences
        input_ids = pad_sequence(all_query_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(all_query_attention, batch_first=True, padding_value=0)
        query_batch_index = torch.tensor(query_batch_index, dtype=torch.int64)
    else:
        # Legacy caption grounding (one text per sample)
        input_ids = pad_sequence(
            [item["dino"]["input_ids"] for item in batch],
            batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = pad_sequence(
            [item["dino"]["attention_mask"] for item in batch],
            batch_first=True, padding_value=0
        )
        for item in batch:
            targets.extend(item["dino"]["targets"])
        query_batch_index = None

    collated = {
        "dino_pixel_values": pixel_values,
        "clip_pixel_values": clip_pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
    }
    if detection_mode:
        collated["query_batch_index"] = query_batch_index  # optional downstream use
    return collated



class TrainingConfig(BaseModel):
    batch_size: ClassVar[int] = 1
    input_channels: ClassVar[int] = 3
    image_h: ClassVar[int] = 224
    image_w: ClassVar[int] = 224
    steps: ClassVar[int] = 0
    epochs: ClassVar[int] = 10
    lr: ClassVar[float] = 2e-4
    accumulation_steps: ClassVar[int] = 4
    number_of_items : ClassVar[int] = 80000
    caption_len : ClassVar[int] = 20 
    save_every : ClassVar[int] = 1000
    checkpoint_path : ClassVar[str] = "checkpoint.pth"



dotenv.load_dotenv()


def Freeze_model(clip_encoder, dino_model):
    for name, param in dino_model.named_parameters():
        if "query_refiner" not in name and "clip" not in name:
            param.requires_grad = False
    
    for name, param in clip_encoder.named_parameters():
        param.requires_grad = False

    return clip_encoder, dino_model


@contextmanager
def temporary_queries(model, new_queries):
    # Make sure we're accessing the right location for query embeddings
    if hasattr(model, "query_position_embeddings"):
        query_pos_embed = model.query_position_embeddings
    elif hasattr(model, "model") and hasattr(model.model, "query_position_embeddings"):
        query_pos_embed = model.model.query_position_embeddings
    else:
        raise AttributeError("Could not find query_position_embeddings in model")
    
    # Clone the original weights
    old = query_pos_embed.weight.data.clone()
    
    # Debug info
    print(f"Original query shape: {old.shape}")
    print(f"New queries shape: {new_queries.shape}")
    
    # Make sure new_queries has the right shape (remove batch dimension if present)
    if new_queries.dim() == 3:  # [B, Q, D]
        new_queries = new_queries.squeeze(0)  # Remove batch dimension -> [Q, D]
    
    with torch.no_grad():
        try:
            query_pos_embed.weight.copy_(new_queries)
        except Exception as e:
            print(f"Error during weight copying: {e}")
            print(f"Target shape: {query_pos_embed.weight.shape}, Source shape: {new_queries.shape}")
            raise
    try:
        yield
    finally:
        # Restore original weights
        with torch.no_grad():
            query_pos_embed.weight.copy_(old)



class CLIPVisionEncoder(nn.Module):
    def __init__(self, model_name="clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.vision = self.clip.vision_model
       
    def forward(self, clip_pixel_values):
        pixel_values = clip_pixel_values.to(self.clip.device)
        vision_outputs = self.vision(pixel_values, output_hidden_states=False)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # drop CLS
        return patch_tokens  # [B, P, hidden]



class GroundingDINOLocalizer(nn.Module):

    """
    GroundingDINO localizer with direct query embedding passing.
    
    This implementation directly passes enriched query embeddings to the model's forward method
    using the 'query_embeds' parameter instead of modifying weights in-place, making it
    gradient-safe and more robust.
    """

    def __init__(self, model_name="grounding_dino_tiny_local"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("grounding_dino_tiny_local")
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_name)

        # Determine DINO query embedding dimension
        self.query_embed = self.get_query_embed_module() 
        self.dino_query_dim = self.query_embed.weight.shape[-1]

        # Multi-head attention refiner (two layers)
        self.query_refiner = MHAQueryRefiner(self.dino_query_dim, patch_dim=768, num_heads=8, depth=2)

        print(f"DINO query embedding dimension: {self.dino_query_dim}")

    
    def get_query_embed_module(self):
        # First check for query_position_embeddings in model or model.model
        query_embed_module = getattr(self.model, "query_position_embeddings", None)
        if query_embed_module is None:
            query_embed_module = getattr(getattr(self.model, "model", object()), "query_position_embeddings", None)
        
        # If not found, look in model.decoder or model.model.decoder
        if query_embed_module is None and hasattr(self.model, "decoder"):
            query_embed_module = getattr(self.model.decoder, "query_embed", None)
        if query_embed_module is None and hasattr(getattr(self.model, "model", object()), "decoder"):
            query_embed_module = getattr(self.model.model.decoder, "query_embed", None)
            
        # If still not found, check base_model
        if query_embed_module is None and hasattr(self.model, "base_model"):
            query_embed_module = getattr(self.model.base_model, "query_position_embeddings", None)
        
        if query_embed_module is None:
            raise AttributeError("Could not locate query embeddings in GroundingDINO model for integration.")
        
        print(f"Found query embeddings at {type(query_embed_module).__name__} with shape {query_embed_module.weight.shape}")
        return query_embed_module

    
    def get_transformer_decoder(self):
        """
        Extract the transformer decoder from the model to directly pass query embeddings
        """
        # Try to find the transformer component that uses the query embeddings
        if hasattr(self.model, "transformer"):
            return self.model.transformer
        elif hasattr(self.model, "model") and hasattr(self.model.model, "transformer"):
            return self.model.model.transformer
        raise AttributeError("Could not locate transformer in GroundingDINO model")


    def visualize_query_attention(self, patch_feats, query_indices=None, images=None):
        """
        Visualize how different queries attend to different parts of the image
        
        Args:
            patch_feats: CLIP patch features [B, P, 768]
            query_indices: Optional list of query indices to visualize
            images: Optional list of original images for reference
        """
        # Process just first batch element for visualization
        patches_b = patch_feats[0].unsqueeze(0) if patch_feats.dim() > 2 else patch_feats.unsqueeze(0)
        base_queries = self.query_embed.weight.data.unsqueeze(0)  # [1, Q, D]
        
        # Get visualization from the MHAQueryRefiner
        fig = self.query_refiner.visualize_attention(base_queries, patches_b, query_indices)
        return fig

    def forward(self, patch_feats, texts=None, images=None, debug=False):
        """
        patch_feats: CLIP patch features [B, P, 768]
        texts: list of strings
        images: list of PIL images (for processor)
        debug: if True, returns additional debug info including attention maps
        """
        if texts and images:
            inputs = self.processor(images=images, text=texts, return_tensors="pt").to(patch_feats.device)
        elif images:
            inputs = self.processor(images=images, return_tensors="pt").to(patch_feats.device)
        else:
            raise ValueError("Need images (and optionally text).")

        base_queries = self.query_embed.weight.data  # [Q, D]
        outputs_list = []
        debug_info = []

        B = patch_feats.size(0)
        for b in range(B):
            patches_b = patch_feats[b].unsqueeze(0)  # [1,P,768]
            
            # Get enriched queries and attention maps for visualization
            if debug:
                enriched, attn_maps = self.query_refiner(
                    base_queries.unsqueeze(0), patches_b, return_attn=True
                )
                debug_info.append({"attention_maps": attn_maps})
            else:
                enriched = self.query_refiner(base_queries.unsqueeze(0), patches_b)
            
            # Create query embeddings for this specific batch element
            # Instead of modifying weights in-place, pass the enriched queries directly
            fused_queries = enriched  # Shape: [1, Q, D]
            
            # Process inputs for this batch element
            single_inputs = {k: v[b].unsqueeze(0) for k, v in inputs.items()}
            
            # Pass the query embeddings directly to the model's forward method
            # Use the temporary_queries context manager to modify the queries
            try:
                with temporary_queries(self.model, fused_queries):
                    out_b = self.model(**single_inputs)
                outputs_list.append(out_b)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                raise

        # Merge results
        ref = outputs_list[0]
        merged = {}
        for k, v in ref.items():
            if torch.is_tensor(v):
                merged[k] = torch.cat([o[k] for o in outputs_list], dim=0)
            else:
                merged[k] = v
                
        result = type(ref)(**merged)
        
        if debug:
            return result, debug_info
        return result
        

class MHAQueryRefiner(nn.Module):
    def __init__(self, query_dim, patch_dim, num_heads=8, depth=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MHALayer(query_dim, patch_dim, num_heads, dropout) for _ in range(depth)
        ])


    def forward(self, queries, patches, return_attn=False):  # queries [B,Q,D], patches [B,P,patch_dim]
        x = queries
        all_attentions = []
        
        for i, layer in enumerate(self.layers):
            # Get attention weights from each layer
            if return_attn:
                x, attn_weights = layer(x, patches, return_attn=True)
                all_attentions.append(attn_weights)
            else:
                x = layer(x, patches)
        
        if return_attn:
            return x, all_attentions
        return x
        
    def visualize_attention(self, queries, patches, query_indices=None, figsize=(15, 10)):
        """
        Visualize attention maps for specific queries
        
        Args:
            queries: Input query embeddings [B, Q, D]
            patches: CLIP patch embeddings [B, P, D]
            query_indices: Optional list of query indices to visualize (default: first 4)
            figsize: Figure size for matplotlib
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        import math
        
        # Forward pass with attention collection
        _, attentions = self.forward(queries, patches, return_attn=True)
        
        # Default to first 4 queries if not specified
        if query_indices is None:
            query_indices = list(range(min(4, queries.shape[1])))
        
        # Extract from batch dimension 0
        attentions = [attn[0].detach().cpu().numpy() for attn in attentions]  # List of [Q, P]
        
        num_layers = len(attentions)
        num_queries = len(query_indices)
        
        # Create figure grid
        fig, axes = plt.subplots(num_queries, num_layers, figsize=figsize)
        if num_queries == 1:
            axes = axes.reshape(1, -1)
        
        # Get patch dimensions (assuming square patches)
        if patches.shape[1] == 196:  # CLIP ViT-B/32 has 14x14 patches
            patch_size = 14
        elif patches.shape[1] == 50:  # Some models have 10x5 patches
            patch_size = (5, 10)
        else:
            patch_size = int(math.sqrt(patches.shape[1]))
            
        # Set a title for the whole figure
        fig.suptitle(f"Attention Maps for {num_queries} Queries Across {num_layers} Layers", fontsize=16)
        
        # Plot attention maps
        for q_idx, query_idx in enumerate(query_indices):
            for l_idx in range(num_layers):
                ax = axes[q_idx, l_idx]
                
                # Get attention weights for this query at this layer
                attn = attentions[l_idx][query_idx]  # [P]
                
                # Reshape to 2D grid based on patch layout
                if isinstance(patch_size, tuple):
                    attn_map = attn.reshape(patch_size)
                else:
                    # Handle case where patch count isn't a perfect square
                    try:
                        attn_map = attn.reshape(patch_size, patch_size)
                    except:
                        # Just display as-is if reshaping fails
                        attn_map = attn.reshape(1, -1)
                    
                # Plot heatmap
                im = ax.imshow(attn_map, cmap='viridis')
                ax.set_title(f"Query {query_idx}, Layer {l_idx+1}")
                ax.axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        return fig
        

class MHALayer(nn.Module):
    def __init__(self, query_dim, patch_dim, num_heads, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_p = nn.LayerNorm(patch_dim)
        self.mha = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.kv_proj = nn.Linear(patch_dim, query_dim)  # unify patch dim to query dim
        self.ff_norm = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, patches, return_attn=False):
        q_norm = self.norm_q(queries)
        p_norm = self.norm_p(patches)
        kv = self.kv_proj(p_norm)  # [B,P,D]
        
        # Get attention weights for each query with patches
        attn_out, attn_weights = self.mha(q_norm, kv, kv, need_weights=True)
        # attn_weights shape: [B, Q, P] - batch, queries, patches
        
        x = queries + self.dropout(attn_out)
        x = x + self.ff(self.ff_norm(x))
        
        if return_attn:
            return x, attn_weights
        return x


class CLIPPatchGroundingDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionEncoder()
        self.localizer = GroundingDINOLocalizer()
        
        # Move models to the same device if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, images, texts=None):
        patch_feats = self.clip(images)
        results = self.localizer(patch_feats, texts=texts, images=images)
        return results


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


def main():
    image = Image.open("/Users/preetamverma/Desktop/image_cap_model_test_images/gettyimages-144103223-2048x2048.jpg")

    model = CLIPPatchGroundingDINO().eval()

    clip_encoder, dino_model = Freeze_model(model.clip, model)

    model.clip = clip_encoder 
    model = dino_model 


    from utils import setup_data
    from datasetlite import DataLoaderLite 

    device = torch.device("mps")


    train_dataset, val_dataset, id_to_name, name_to_id  = setup_data(TrainingConfig.number_of_items)
    train_dataset_cocooptions = DataLoaderLite(train_dataset, caption_length=TrainingConfig.caption_len)
    train_dataset_cocooptions = DataLoaderLite(val_dataset, caption_length=TrainingConfig.caption_len)

    train_dataloader = DataLoader(train_dataset_cocooptions, batch_size=TrainingConfig.batch_size, collate_fn=collate_fn, shuffle=True)

    model = GroundingDinoForObjectDetection.from_pretrained("grounding_dino_tiny_local")

    model = model.to(device)

    for batch in train_dataloader:
        # Only move tensor fields; ignore list/str (e.g. query_names)
        cleaned_labels = []
        for t in batch["targets"]:
            ld = {}
            for k, v in t.items():
                if k in {"boxes", "class_labels", "positive_map", "query_ids"}:  # tensors we care about
                    if hasattr(v, "to"):
                        ld[k] = v.to(device)
                # skip query_names (list of strings) and other non-tensor metadata
            cleaned_labels.append(ld)

        outputs = model(
            pixel_values=batch["dino_pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=cleaned_labels,
        )

        print ("=*="*60)
        print (outputs.logits.shape)
        print ("cleaned_labels", cleaned_labels)

        break

    LLLL


    device = torch.device("mps")


    print(f"Using device: {device}")


    
 
    
    texts = ["a train"]

    # Process the image with the CLIP encoder
    with torch.no_grad():
        clip_patches = model.clip([image])
        print(f"\n\t clip_patches {clip_patches.shape}")
        
        # Run object detection with debug info
        outputs, debug_info = model.localizer(clip_patches, texts=texts, images=[image], debug=True)
        
        # Visualize attention for first few queries
        fig = model.localizer.visualize_query_attention(clip_patches, query_indices=[0, 1, 2, 3])
        fig.savefig("query_attention_visualization.png")
        print("Saved attention visualization to query_attention_visualization.png")

    # Postprocess detections
    processor = model.localizer.processor
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        print(f"{label}: {score:.2f} {box.tolist()}")
    
    visualize_boxes(image, results["boxes"], results["labels"], results["scores"]) 


main()