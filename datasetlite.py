from torch.utils.data import Dataset
from transformers import CLIPProcessor
from dotenv import load_dotenv 
import torch 


load_dotenv()

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)


class DataLoaderLite(Dataset):

    tokenizer = None 

    def __init__(self, train_dataset_cocooptions, coco_detection_dataset, caption_length=20, num_img_tokens=64, tokenizer=tokenizer):
        self.train_dataset_cocooptions = train_dataset_cocooptions
        self.det_ds = coco_detection_dataset
        self.caption_length = caption_length
        self.tokenizer = tokenizer 
        self.num_img_tokens = num_img_tokens


    def __len__(self):
        return len(self.det_ds)
    
    def __getitem__(self, idx):
        img, image_captions = self.train_dataset_cocooptions[idx]
        img_d, det_target = self.det_ds[idx]

         # Apply CLIPProcessor
        image_tensor = clip_processor(images=img, return_tensors="pt")
        image_tensor = image_tensor["pixel_values"].squeeze(0)

        # prepend <START>, append <END>
        caption = "<START> " + image_captions[0] + " <END>"

        # --- BBoxes and Class Labels ---
        if len(det_target) == 0:
            # No objects detected
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64) - 1  # -1 indicates no label
        else:
            boxes = torch.tensor([ann['bbox'] for ann in det_target], dtype=torch.float32)
            labels = torch.tensor([ann['category_id'] for ann in det_target], dtype=torch.int64)

            # Normalize [x, y, w, h] to [0,1]
            img_w, img_h = img_d.size
            boxes[:, 0] /= img_w
            boxes[:, 1] /= img_h
            boxes[:, 2] /= img_w
            boxes[:, 3] /= img_h

        # --- Objectness ---
        num_objects = boxes.shape[0]
        objectness = torch.zeros(self.num_img_tokens, 1)
        objectness[:num_objects] = 1.0

        # tokenize with GPT2 tokenizer
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            max_length=self.caption_length,
            padding="max_length",
            truncation=True
        )
        # Pad bbox and class
        if num_objects < self.num_img_tokens:
            pad_len = self.num_img_tokens - num_objects
            boxes = torch.cat([boxes, torch.zeros(pad_len, 4)], dim=0)
            labels = torch.cat([labels, torch.full((pad_len,), -1, dtype=torch.int64)], dim=0)
        else:
            boxes = boxes[:self.num_img_tokens]
            labels = labels[:self.num_img_tokens]

        return {
            "image": image_tensor,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "bboxes": boxes,
            "class_labels": labels,
            "objectness": objectness
        }