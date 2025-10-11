from torch.utils.data import Dataset
from transformers import CLIPProcessor
from dotenv import load_dotenv 
import torch 


load_dotenv()

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)


class DataLoaderLite(Dataset):

    tokenizer = None 

    def __init__(self, train_dataset_cocooptions, caption_length=20, tokenizer=tokenizer):
        self.train_dataset_cocooptions = train_dataset_cocooptions
        self.caption_length = caption_length
        self.tokenizer = tokenizer 

    def __len__(self):
        return len(self.train_dataset_cocooptions)
    
    def __getitem__(self, idx):
        img, image_captions = self.train_dataset_cocooptions[idx]

         # Apply CLIPProcessor
        image_tensor = clip_processor(images=img, return_tensors="pt")
        image_tensor = image_tensor["pixel_values"].squeeze(0)

        # prepend <START>, append <END>
        caption = "<START> " + image_captions[0] + " <END>"

        # tokenize with GPT2 tokenizer
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            max_length=self.caption_length,
            padding="max_length",
            truncation=True
        )

        return image_tensor, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)