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
for img_tensor, input_ids, attn_mask in test_loader:
    print("Image shape:", img_tensor.shape)
    print("Token IDs:", input_ids)
    print("Attention mask:", attn_mask)
    break
