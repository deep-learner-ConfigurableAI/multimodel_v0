from torch.utils.data import Dataset
from transformers import CLIPProcessor
from dotenv import load_dotenv 
import torch 
import json 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor
from transformers.models.grounding_dino import GroundingDinoForObjectDetection

load_dotenv()

clip_processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32", use_fast=True)
dino_processor = AutoProcessor.from_pretrained("grounding_dino_tiny_local", use_fast=True)



with open("label_dict.json", "r") as f:
    label_dict = json.loads(f.read()) 

# Raw COCO id -> name mapping (already in file)
id_to_name = label_dict["label_dict"]

"""Simplified label handling: we now use raw COCO category_id values directly
without remapping to a contiguous space. The classifier head is sized to cover
the max raw id (e.g. up to 90). Padding uses -1 which is ignored in loss.
"""



class DataLoaderLite(Dataset):
    def __init__(
        self,
        train_dataset,
        caption_length=20,
        text_grounded=True,
        debug=False,
        detection_only=True,
        query_mode="present",  # "present" | "all"
        num_random_negatives=0,  # sample absent categories (only if query_mode=="present")
        use_all_categories_when_empty=True,
    ):
        self.train_dataset = train_dataset
        self.caption_length = caption_length
        self.text_grounded = text_grounded
        self.debug = debug
        self.detection_only = detection_only
        self.query_mode = query_mode
        self.num_random_negatives = num_random_negatives
        self.use_all_categories_when_empty = use_all_categories_when_empty

        # Precompute sorted full category name list (id_to_name keys are strings)
        self.all_category_items = sorted(
            [(int(k), v) for k, v in id_to_name.items() if v is not None]
        )
        self.all_category_ids = [cid for cid, _ in self.all_category_items]
        self.all_category_names = [name for _, name in self.all_category_items]
        
    def __len__(self):
        return len(self.train_dataset)

    def xywh_to_cxcywh(self, box, img_w, img_h):
        x_min, y_min, w, h = box
        cx = x_min + w / 2
        cy = y_min + h / 2
        return [cx / img_w, cy / img_h, w / img_w, h / img_h]

    def __getitem__(self, idx):
        img, image_captions, det_target, orig_h, orig_w = self.train_dataset[idx]

        # Choose caption (used only if not detection_only)
        caption = image_captions[0] if isinstance(image_captions, (list, tuple)) else image_captions

        # Build query phrases for detection-only mode
        if self.detection_only:
            if len(det_target) == 0 and self.use_all_categories_when_empty:
                # Fallback: use full category list (no positives, image may be skipped downstream)
                present_ids = []
                query_names = self.all_category_names
                query_ids = self.all_category_ids
            else:
                present_ids = sorted({ann['category_id'] for ann in det_target})
                if self.query_mode == "all":
                    query_ids = self.all_category_ids
                    query_names = self.all_category_names
                else:  # present
                    query_ids = list(present_ids)
                    query_names = [id_to_name.get(str(cid), "") for cid in query_ids]
                    # Optional negative sampling (absent categories)
                    if self.num_random_negatives > 0:
                        import random
                        absent = [cid for cid in self.all_category_ids if cid not in present_ids]
                        random.shuffle(absent)
                        neg_sample = absent[: self.num_random_negatives]
                        query_ids.extend(neg_sample)
                        query_names.extend([id_to_name.get(str(cid), "") for cid in neg_sample])

            # Processor can accept list of strings
            processed = dino_processor(images=img, text=query_names, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)
            input_ids = processed["input_ids"].squeeze(0)
            attention_mask = processed["attention_mask"].squeeze(0)

            # Map each annotation category_id -> index within query_ids for class_labels
            catid_to_query_index = {cid: i for i, cid in enumerate(query_ids)}
        else:
            # Original caption-driven grounding (less suitable for pure detection)
            processed = dino_processor(images=img, text=caption, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)
            input_ids = processed["input_ids"].squeeze(0)
            attention_mask = processed["attention_mask"].squeeze(0)

        clip_image_tensor = clip_processor(images=img, return_tensors="pt")
        clip_pixel_values = clip_image_tensor["pixel_values"].squeeze(0)

        # Boxes / labels (handle empty)
        if len(det_target) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            if self.detection_only:
                # No boxes, still produce empty labels referencing queries
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor([
                self.xywh_to_cxcywh(ann['bbox'], orig_w, orig_h) for ann in det_target
            ], dtype=torch.float32)
            if self.detection_only:
                # Use index within queries
                labels = torch.tensor([
                    catid_to_query_index[ann['category_id']] for ann in det_target
                ], dtype=torch.int64)
            else:
                # Raw category ids (legacy behavior)
                labels = torch.tensor([ann['category_id'] for ann in det_target], dtype=torch.int64)

        # Clamp boxes to [0,1] to avoid invalid cost matrix values
        if boxes.numel() > 0:
            boxes = boxes.clamp(0.0, 1.0)

        # Validate numeric integrity
        if self.debug:
            if boxes.numel() > 0 and (not torch.isfinite(boxes).all()):
                print(f"[DEBUG] Non-finite values in boxes at idx {idx}: {boxes}")
            if labels.numel() > 0 and (not torch.isfinite(labels.float()).all()):
                print(f"[DEBUG] Non-finite values in labels at idx {idx}: {labels}")

        # DINO/DETR expects targets as list of dicts
        targets = []
        target_dict = {"boxes": boxes, "class_labels": labels}

        if self.detection_only:
            # Provide mapping for downstream use / debugging
            target_dict["query_names"] = query_names
            target_dict["query_ids"] = torch.tensor(query_ids, dtype=torch.int64)

        # Optionally add positive_map for text-grounded training (currently disabled)
        # if self.text_grounded and len(caption) > 0 and boxes.shape[0] > 0:
        #     words = caption.split()
        #     positive_map = torch.zeros((len(words), boxes.shape[0]))
        #     for i, word in enumerate(words):
        #         for j, label_id in enumerate(labels.tolist()):
        #             label_name = id_to_name.get(str(label_id), "")
        #             if word.lower() in label_name.lower():
        #                 positive_map[i, j] = 1.0
        #     target_dict["positive_map"] = positive_map

        targets.append(target_dict)

        batch = {
            "dino": {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "targets": targets,
            },
            "clip": {
                "pixel_values": clip_pixel_values,
            },
        }

        if self.debug:
            mode = "DET" if self.detection_only else "CAP"
            print(
                f"[DEBUG] idx={idx} mode={mode} boxes={boxes.shape} labels={labels.shape} tokens={input_ids.shape[-1]} present_ids={len(det_target)}"
            )

        return batch