from transformers import CLIPModel
import torch.nn as nn 
import torch 
import torch.nn.functional as F
import math

class DeformableAttention(nn.Module):
    """
    Deformable Attention module similar to Deformable DETR
    """
    def __init__(self, dim, num_heads=8, num_points=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize offsets to be small
        nn.init.constant_(self.sampling_offsets.weight, 0)
        nn.init.uniform_(self.sampling_offsets.bias, -0.1, 0.1)
        
    def forward(self, query, reference_points, input_features, input_spatial_shapes):
        """
        Args:
            query (Tensor): [B, L_q, C]
            reference_points (Tensor): [B, L_q, n_levels, 2], normalized coordinates (0,1)
            input_features (List[Tensor]): List of features from each scale
            input_spatial_shapes (Tensor): [n_levels, 2], (H, W) of each level
        """
        B, L_q, _ = query.shape
        n_levels = len(input_features)
        
        # Project value and reshape for multi-head attention
        value = torch.cat(input_features, dim=1)  # [B, L_v, C]
        value = self.value_proj(value).reshape(B, -1, self.num_heads, self.head_dim)
        
        # Compute sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(
            B, L_q, self.num_heads, self.num_points, 2)
        
        # Compute attention weights
        attention_weights = self.attention_weights(query).view(
            B, L_q, self.num_heads, self.num_points)
        attention_weights = F.softmax(attention_weights, -1)
        
        # Process each level
        output = torch.zeros_like(query)
        
        # Convert reference points to absolute coordinates for each level
        start_idx = 0
        for lvl, (feat, spatial_shape) in enumerate(zip(input_features, input_spatial_shapes)):
            H, W = spatial_shape
            L_v = H * W
            
            # Get reference points with offsets
            ref_points_lvl = reference_points[:, :, lvl:lvl+1, :]  # [B, L_q, 1, 2]
            ref_points_lvl = ref_points_lvl.repeat(1, 1, self.num_heads, 1)  # [B, L_q, num_heads, 2]
            
            # Add learned offsets to reference points
            sample_points = ref_points_lvl[:, :, :, None, :] + sampling_offsets[:, :, :, :, :]
            
            # Clamp to [0, 1]
            sample_points = torch.clamp(sample_points, 0, 1)
            
            # Convert normalized coordinates to pixel coordinates
            sample_points_x = sample_points[:, :, :, :, 0] * (W - 1)
            sample_points_y = sample_points[:, :, :, :, 1] * (H - 1)
            
            # Bilinear interpolation
            sample_points_x0 = torch.floor(sample_points_x).long()
            sample_points_x1 = torch.min(sample_points_x0 + 1, torch.tensor(W - 1))
            sample_points_y0 = torch.floor(sample_points_y).long()
            sample_points_y1 = torch.min(sample_points_y0 + 1, torch.tensor(H - 1))
            
            # Get the four nearest points
            # Reshape feat to [B, H, W, C]
            feat_reshape = feat.view(B, H, W, -1)
            
            # Gather the four nearest points
            idx_y0x0 = sample_points_y0 * W + sample_points_x0
            idx_y0x1 = sample_points_y0 * W + sample_points_x1
            idx_y1x0 = sample_points_y1 * W + sample_points_x0
            idx_y1x1 = sample_points_y1 * W + sample_points_x1
            
            # Get values at those indices
            # This is a simplified approximation - in practice, use grid_sample
            # for accurate bilinear interpolation
            value_lvl = value[:, start_idx:start_idx+L_v].reshape(B, H, W, self.num_heads, self.head_dim)
            
            # Linear combination of the values with attention weights
            weighted_values = attention_weights[:, :, :, lvl].unsqueeze(-1) * \
                              torch.gather(value_lvl, dim=1, index=idx_y0x0.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim))
            
            output += weighted_values.reshape(B, L_q, self.dim)
            
            # Update start index for next level
            start_idx += L_v
        
        # Final projection
        output = self.output_proj(output)
        return self.dropout(output)


class MultiScaleFeaturesExtractor(nn.Module):
    """Extract multi-scale features from CLIP ViT output"""
    def __init__(self, input_dim, embed_dim, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Projections for different scales
        self.projections = nn.ModuleList([
            nn.Linear(input_dim, embed_dim),  # Scale 1 (original)
            nn.Linear(input_dim * 4, embed_dim),  # Scale 2 (2x2 patches)
            nn.Linear(input_dim * 9, embed_dim),  # Scale 3 (3x3 patches)
        ])
        
    def reshape_features(self, features, H, W):
        """Reshape token features back to spatial dimensions"""
        B = features.shape[0]
        return features.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
    def forward(self, features, img_size):
        """
        Extract features at multiple scales
        Args:
            features: [B, N, D] ViT patch features
            img_size: (H, W) original image size
        """
        B, N, D = features.shape
        
        # Calculate grid size
        grid_h = img_size[0] // self.patch_size
        grid_w = img_size[1] // self.patch_size
        
        # Reshape to spatial dimensions
        spatial_features = features.reshape(B, grid_h, grid_w, D)
        
        multi_scale_features = []
        spatial_shapes = []
        
        # Scale 1: Original resolution
        scale1 = self.projections[0](features)  # [B, N, embed_dim]
        multi_scale_features.append(scale1)
        spatial_shapes.append(torch.tensor([grid_h, grid_w]))
        
        # Scale 2: 2x2 patch grouping
        # Group 2x2 neighboring patches
        if grid_h >= 2 and grid_w >= 2:
            # Use adaptive avgpool for even grid sizes
            scale2_spatial = F.adaptive_avg_pool2d(
                spatial_features.permute(0, 3, 1, 2), 
                (grid_h // 2, grid_w // 2)
            ).permute(0, 2, 3, 1)
            
            scale2_flat = scale2_spatial.reshape(B, -1, D)
            scale2 = self.projections[1](scale2_flat)  # [B, N/4, embed_dim]
            multi_scale_features.append(scale2)
            spatial_shapes.append(torch.tensor([grid_h // 2, grid_w // 2]))
        
        # Scale 3: 3x3 patch grouping or global pooling for smaller images
        if grid_h >= 3 and grid_w >= 3:
            # Use adaptive avgpool for arbitrary grid sizes
            scale3_spatial = F.adaptive_avg_pool2d(
                spatial_features.permute(0, 3, 1, 2),
                (max(1, grid_h // 3), max(1, grid_w // 3))
            ).permute(0, 2, 3, 1)
            
            scale3_flat = scale3_spatial.reshape(B, -1, D)
            scale3 = self.projections[2](scale3_flat)  # [B, N/9, embed_dim]
            multi_scale_features.append(scale3)
            spatial_shapes.append(torch.tensor([max(1, grid_h // 3), max(1, grid_w // 3)]))
        
        return multi_scale_features, torch.stack(spatial_shapes)


class ReferencePointsGenerator(nn.Module):
    """Generate reference points for deformable attention"""
    def __init__(self, embed_dim, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_points = num_points
        self.point_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2)  # (x, y) normalized coordinates
        )
        
    def forward(self, query_features, num_levels):
        """
        Generate reference points for deformable attention
        Args:
            query_features: [B, L_q, D]
            num_levels: number of feature levels to attend to
        Returns:
            reference_points: [B, L_q, num_levels, 2] normalized (0,1)
        """
        B, L_q, _ = query_features.shape
        
        # Predict reference points
        ref_points = self.point_predictor(query_features)  # [B, L_q, 2]
        ref_points = torch.sigmoid(ref_points)  # Normalize to [0, 1]
        
        # Expand to all levels
        ref_points = ref_points.unsqueeze(2).expand(-1, -1, num_levels, -1)
        
        return ref_points


class MultiScaleCLIPEncoder(nn.Module):
    def __init__(
        self, 
        embed_size,
        model_name="openai/clip-vit-base-patch32", 
        freeze_vision=True,
        num_heads=8,
        num_points=4,
        dropout=0.1
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.vision = self.clip.vision_model  # Vision tower (ViT)
        self.patch_size = self.vision.config.patch_size
        self.hidden_size = self.vision.config.hidden_size
        self.embed_size = embed_size

        if freeze_vision:
            for param in self.vision.parameters():
                param.requires_grad = False

        # Multi-scale feature extractor
        self.multi_scale_extractor = MultiScaleFeaturesExtractor(
            self.hidden_size, embed_size, patch_size=self.patch_size
        )
        
        # Query embeddings that will attend to different scales
        self.query_embeddings = nn.Parameter(torch.randn(1, 64, embed_size))
        
        # Reference points generator
        self.reference_generator = ReferencePointsGenerator(embed_size, num_points)
        
        # Deformable attention module
        self.deformable_attn = DeformableAttention(
            embed_size, num_heads=num_heads, num_points=num_points, dropout=dropout
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, x):
        """
        x: (B, 3, H, W), raw images expected to be normalized 
        Returns: (B, num_queries, embed_size) - features with multi-scale information
        """
        B = x.shape[0]
        img_size = (x.shape[2], x.shape[3])
        
        # Get CLIP's vision features
        outputs = self.vision(pixel_values=x, output_hidden_states=False)
        feats = outputs.last_hidden_state  # (B, num_patches+1, D) 

        # Optionally drop CLS token (index 0), keep patch tokens
        patch_feats = feats[:, 1:, :]  # (B, num_patches, D)
        
        # Extract multi-scale features
        multi_scale_features, spatial_shapes = self.multi_scale_extractor(patch_feats, img_size)
        
        # Expand query embeddings to batch size
        queries = self.query_embeddings.expand(B, -1, -1)
        
        # Generate reference points for deformable attention
        reference_points = self.reference_generator(queries, len(multi_scale_features))
        
        # Apply deformable attention
        attended_features = self.deformable_attn(
            queries, reference_points, multi_scale_features, spatial_shapes
        )
        
        # Final projection
        output = self.final_proj(attended_features)
        
        return output
