import os
os.environ["ARROW_USER_SIMD_LEVEL"] = "none"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"



import torch 
from setup_model import get_models

from datasetlite import DataLoaderLite 
from torch.utils.data import DataLoader
from PIL import Image
import os, sys 

import torch
from torch.nn import functional as F
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, RepetitionPenaltyLogitsProcessor
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import matplotlib.pyplot as plt

from generate import generate_caption, visualize_caption 


device = torch.device("mps")




torch.set_num_threads(1)
torch.set_num_interop_threads(1)



TrainingConfig, encoder_model, decoder_model , pad_token_id, tokenizer, extras_dict = get_models() 



def load_images_from_folder(folder_path, tokenizer, dummy_caption="A photo.", max_images=None):
    """
    Loads all images from a folder and wraps them into DataLoaderLite.
    If captions are not available, uses a dummy caption.
    """
    image_list = []
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for i, file_name in enumerate(os.listdir(folder_path)):
        if max_images and i >= max_images:
            break
        if os.path.splitext(file_name)[1].lower() not in supported_ext:
            continue

        img_path = os.path.join(folder_path, file_name)
        try:
            img = Image.open(img_path).convert("RGB")
            # You can replace dummy_caption with something dynamic if needed
            image_list.append((img, [dummy_caption]))
        except Exception as e:
            print(f" Skipping {file_name}: {e}")

    dataset = DataLoaderLite(image_list, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=1, shuffle=False)




# create dataloader from a local folder
folder_path = "/Users/preetamverma/Desktop/image_cap_model_test_images"
test_loader = load_images_from_folder(folder_path, tokenizer)

# Iterate over the test loader
for image_tensor, caption_tensor, attention_mask in test_loader:
    print("Image shape:", image_tensor.shape)
    print("Token IDs:", caption_tensor)
    print("Attention mask:", attention_mask)

    image_tensor, caption_tensor, attention_mask = image_tensor.to(device), caption_tensor.to(device), attention_mask.to(device)
    B, C, H, W = image_tensor.shape
    gt_caption = tokenizer.decode(caption_tensor[0].tolist(), skip_special_tokens=True) 
    print("Ground Truth:\t", gt_caption)

    # Generate a caption
    caption, generated_ids, similarity_logs = generate_caption(
        image_tensor,
        encoder_model,
        decoder_model,
        tokenizer,
        device,
        temperature=0.7,  # Lower for less randomness
        repetition_penalty=1.5,  # Higher to avoid repetition,
        debug_similarity=True,
        use_image=True
    )
    # Visualize the result
    visualize_caption(image_tensor, caption, gt_caption)
    break 
