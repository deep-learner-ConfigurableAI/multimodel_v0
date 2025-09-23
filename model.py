import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import BertModel, BertTokenizer
import math
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


TYPE_MAP = {
    "RECTANGLE": 0,
    "TEXT": 1,
    "GROUP": 2,
    "VECTOR": 3,
    "TAB": 4,
    "ACTIVE_TAB": 5,
    "TABLE_VIEW": 6
}
NUM_CLASSES = len(TYPE_MAP)


#Node → Feature Vector
def process_node(node, img_w, img_h, depth=0):
    type_id = TYPE_MAP.get(node.get("type", "RECTANGLE"), 0)
    bbox = node.get("absoluteBoundingBox", {"x":0,"y":0,"width":0,"height":0})
    x = bbox["x"] / img_w
    y = bbox["y"] / img_h
    w = bbox["width"] / img_w
    h = bbox["height"] / img_h
    depth_feat = depth / 10.0
    feature = [type_id, x, y, w, h, depth_feat]
    features = [feature]
    for child in node.get("children", []):
        features += process_node(child, img_w, img_h, depth + 1)
    return features


def enrich_with_annotations(figma_node, annotations):
    for ann in annotations.get("annotations", []):
        if figma_node["id"] in ann.get("layer_ids", []):
            figma_node["type"] = ann["type"]
            if "active_layer_id" in ann and figma_node["id"] == ann["active_layer_id"]:
                figma_node["type"] = "ACTIVE_TAB"
    return figma_node



def load_figma_metadata(figma_json_path, annotations=None, img_size=(256,256)):
    with open(figma_json_path, "r") as f:
        figma_data = json.load(f)
    img_w, img_h = img_size
    features = []
    for node in figma_data.get("document", {}).get("children", []):
        if annotations:
            node = enrich_with_annotations(node, annotations)
        features += process_node(node, img_w, img_h)
    return features



def pad_sequence(features_list, max_len=None, pad_value=0.0):
    B = len(features_list)
    feature_dim = len(features_list[0][0])
    if max_len is None:
        max_len = max(len(seq) for seq in features_list)
    padded = torch.full((B, max_len, feature_dim), pad_value, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, seq in enumerate(features_list):
        L = len(seq)
        padded[i, :L, :] = torch.tensor(seq, dtype=torch.float32)
        mask[i, :L] = 1
    return padded, mask


#Image Loading & Transform
image_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return image_transforms(img)


def collate_fn(batch):
    images, metadata_seqs, class_labels, bbox_labels = zip(*batch)
    images = torch.stack(images, dim=0)
    padded_metadata, mask = pad_sequence(metadata_seqs)
    padded_classes, _ = pad_sequence([ [[c] for c in seq] for seq in class_labels ])
    padded_classes = padded_classes.squeeze(-1).long()
    padded_bbox, _ = pad_sequence(bbox_labels)
    return images, padded_metadata, padded_classes, padded_bbox, mask



class FigmaUIDataset(Dataset):
    def __init__(self, data_dir, annotations_dir=None):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        self.annotations_dir = annotations_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        json_file = self.files[idx]
        figma_path = os.path.join(self.data_dir, json_file)
        image_path = figma_path.replace(".json", ".png")

        annotations = None
        if self.annotations_dir:
            ann_file = os.path.join(self.annotations_dir, json_file)
            if os.path.exists(ann_file):
                with open(ann_file, "r") as f:
                    annotations = json.load(f)

        metadata_seq = load_figma_metadata(figma_path, annotations)
        class_labels = [int(f[0]) for f in metadata_seq]
        bbox_labels = [f[1:5] for f in metadata_seq]
        image = load_image(image_path)

        return image, metadata_seq, class_labels, bbox_labels




# -------------------------
# RMS Normalization from scratch
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        # x: [B, D] or [B, T, D]
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm

# -------------------------
# Grouped Query Attention from scratch
# -------------------------
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_groups):
        super().__init__()
        assert hidden_dim % (num_heads * num_groups) == 0, "hidden_dim must be divisible by heads * groups"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_size = hidden_dim // (num_heads * num_groups)
        
        # Weights for query, key, value, and output
        self.Wq = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.Wk = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.Wv = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.Wo = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

    def forward(self, q_input, k_input, v_input):
        B, T, H = k_input.shape[0], k_input.shape[1], k_input.shape[2]
        
        # Linear projections
        q = q_input @ self.Wq  # [B, H]
        k = k_input @ self.Wk  # [B, T, H]
        v = v_input @ self.Wv  # [B, T, H]
        
        # Split into heads and groups
        q = q.view(B, self.num_heads, self.num_groups, self.group_size)  # [B, heads, groups, group_size]
        k = k.view(B, T, self.num_heads, self.num_groups, self.group_size).permute(0, 2, 3, 1, 4)  # [B, heads, groups, T, group_size]
        v = v.view(B, T, self.num_heads, self.num_groups, self.group_size).permute(0, 2, 3, 1, 4)  # [B, heads, groups, T, group_size]
        
        # Attention scores: q @ k.transpose
        attn_scores = torch.einsum('bhgd,bhgtd->bhgt', q, k)  # [B, heads, groups, T]
        attn_scores = attn_scores / math.sqrt(self.group_size)
        
        # Softmax over time dimension
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.einsum('bhgt,bhgtd->bhgd', attn_probs, v)  # [B, heads, groups, group_size]
        
        # Merge heads and groups
        context = context.contiguous().view(B, H)
        
        # Output projection
        output = context @ self.Wo  # [B, H]
        
        return output

# -------------------------
# Image Encoder using ResNet
# -------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, embed_dim)
        
    def forward(self, images):
        features = self.feature_extractor(images)
        pooled = self.pool(features).squeeze(-1).squeeze(-1)
        embeddings = self.linear(pooled)
        return embeddings

# -------------------------
# Text Encoder using BERT
# -------------------------
class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(self.bert.config.hidden_size, embed_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        embeddings = self.linear(pooled)
        return embeddings
    

class FigmaEncoeder(nn.Module):




# -------------------------
# Cross Attention Block using RMS Norm and Grouped Query Attention
# -------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups):
        super().__init__()
        self.rms_norm_q = RMSNorm(embed_dim)
        self.rms_norm_k = RMSNorm(embed_dim)
        self.rms_norm_v = RMSNorm(embed_dim)
        self.gqa = GroupedQueryAttention(embed_dim, num_heads, num_groups)

    def forward(self, query, key, value):
        q_norm = self.rms_norm_q(query)
        k_norm = self.rms_norm_k(key)
        v_norm = self.rms_norm_v(value)
        
        output = self.gqa(q_norm, k_norm, v_norm)
        return output

# -------------------------
# Multimodal Model
# -------------------------
class MultiModalTransformer(nn.Module):
    def __init__(self, metadata_dim=512, embed_dim=512, num_heads=8, num_groups=4, num_classes=10):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.cross_attention = CrossAttentionBlock(embed_dim, num_heads, num_groups)

        # Metadata embedding
        self.metadata_fc = nn.Linear(metadata_dim, embed_dim)


        

        # Multitask heads
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.bbox_regressor = nn.Linear(embed_dim, 4)

    def forward(self, images, input_ids, attention_mask):
        img_embed = self.image_encoder(images)
        txt_embed = self.text_encoder(input_ids, attention_mask)
        
        fused = self.cross_attention(txt_embed, img_embed.unsqueeze(1), img_embed.unsqueeze(1))
        
        logits = self.classifier(fused)
        return logits
    


#----
#training loop and loss functions would go here

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiModalTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
classification_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.SmoothL1Loss()
lambda_bbox = 1.0

train_loader = DataLoader(FigmaUIDataset("figma_jsons/", "annotations/"),
                          batch_size=4, shuffle=True, collate_fn=collate_fn)

for epoch in range(5):
    model.train()
    total_loss = 0
    for images, metadata_seq, class_labels, bbox_labels, mask in train_loader:
        images, metadata_seq, class_labels, bbox_labels, mask = images.to(device), metadata_seq.to(device), class_labels.to(device), bbox_labels.to(device), mask.to(device)

        optimizer.zero_grad()
        class_logits, bbox_preds = model(images, metadata_seq, mask)

        B, T, C = class_logits.shape
        class_logits_flat = class_logits.view(B*T,C)
        class_labels_flat = class_labels.view(B*T)
        bbox_preds_flat = bbox_preds.view(B*T,4)
        bbox_labels_flat = bbox_labels.view(B*T,4)

        loss_class = classification_criterion(class_logits_flat, class_labels_flat)
        loss_bbox = bbox_criterion(bbox_preds_flat, bbox_labels_flat)
        loss = loss_class + lambda_bbox * loss_bbox

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

