from transformers import CLIPModel
import torch.nn as nn 
import os 
import torch 
from dotenv import load_dotenv

load_dotenv()



class CLIPEncoder(nn.Module):
    def __init__(self, embed_size, model_name="openai/clip-vit-base-patch32", freeze_vision=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.vision = self.clip.vision_model  # Vision tower (ViT)

        if freeze_vision:
            for param in self.vision.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(self.vision.config.hidden_size, embed_size)
        self.embed_size = embed_size

    def forward(self, x):
        """
        x: (B, 3, H, W), raw images expected to be normalized 
        Returns: (B, num_patches, embed_size)
        """
        # CLIP’s vision forward gives hidden states for each patch
        outputs = self.vision(pixel_values=x, output_hidden_states=False)
        feats = outputs.last_hidden_state  # (B, num_patches+1, D) 

        # Optionally drop CLS token (index 0), keep patch tokens
        feats = feats[:, 1:, :]  # (B, num_patches, D)

        feats = self.proj(feats)  # (B, num_patches, embed_size)
        return feats
