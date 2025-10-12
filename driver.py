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


device = torch.device("mps")




torch.set_num_threads(1)
torch.set_num_interop_threads(1)



TrainingConfig, encoder_model, decoder_model , pad_token_id, tokenizer, extras_dict = get_models() 




def visualize_caption(image_tensor, caption, gt_caption=None, figsize=(10, 8), similarity_data=None):
    import torch, matplotlib.pyplot as plt

    # Create figure with appropriate subplots
    if similarity_data is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    # Convert tensor to numpy for visualization
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().cpu()
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        img = image_tensor.permute(1, 2, 0).numpy()

        # Normalize to [0,1] for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    else:
        img = image_tensor

    # Plot image
    ax1.imshow(img)
    ax1.axis("off")

    title = f"Generated: {caption}"
    if gt_caption:
        title += f"\nGround truth: {gt_caption}"
    ax1.set_title(title, fontsize=12)

    # Plot similarity map if provided
    if similarity_data is not None:
        global_to_query = similarity_data.get("global_to_query")
        if global_to_query is not None:
            im = ax2.imshow(global_to_query.reshape(1, -1), cmap="viridis", aspect="auto")
            ax2.set_title("Global-Query Token Similarity")
            ax2.set_xlabel("Query Token Index")
            fig.colorbar(im, ax=ax2, label="Cosine Similarity")

            stats_text = f"Global-Query Mean: {global_to_query.mean():.4f}\n"
            stats_text += f"Global-Query Max: {global_to_query.max():.4f}\n"
            if "global_to_img" in similarity_data:
                global_to_img = similarity_data["global_to_img"]
                stats_text += f"Global-Image Mean: {global_to_img.mean():.4f}"

            ax2.text(0.05, 0.05, stats_text, transform=ax2.transAxes,
                     fontsize=9, verticalalignment="bottom",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def generate_caption_3(
    image_tensor,
    encoder,
    decoder,
    tokenizer,
    device,
    max_len=30,
    use_image=True,
    temperature=0.6,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.8,
    min_length=5,
    debug_similarity=False,
):
    """
    Generate caption for an image using the encoder-decoder model (sampling version)
    """
    encoder.eval()
    decoder.eval()

    # Enable similarity logging if requested
    if debug_similarity:
        decoder.debug_similarity = True
        decoder.similarity_logs = []

    # Add batch dimension to image if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Encode image
    if use_image:
        x_embed = encoder(image_tensor)
    else:
        x_embed = torch.zeros((1, 49, decoder.embed_size), device=device)

    # Token setup
    start_id = tokenizer.convert_tokens_to_ids("<START>")
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")
    generated_ids = torch.tensor([[start_id]], device=device)

    # Logit warpers and processors
    warpers = LogitsProcessorList([
        TemperatureLogitsWarper(temperature),
        TopKLogitsWarper(top_k),
        TopPLogitsWarper(top_p),
    ])
    processors = LogitsProcessorList([
        RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
        MinLengthLogitsProcessor(min_length, end_token_id),
    ])

    # Autoregressive generation loop (no beam search)
    for step in range(max_len):
        attn_mask = torch.ones(1, generated_ids.size(1), dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = decoder(x_embed, generated_ids, attn_mask, mode="inference")

        next_token_logits = logits[:, -1, :]

        # Apply processors & warpers
        next_token_logits = processors(generated_ids.to("cpu"), next_token_logits.to("cpu"))
        next_token_logits = warpers(generated_ids.to("cpu"), next_token_logits.to("cpu"))

        # Convert to probabilities and sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append token
        generated_ids = torch.cat([generated_ids.to(device), next_token.to(device)], dim=1)

        # Stop if END token generated
        if next_token.item() == end_token_id:
            break

    # Decode caption
    caption = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)

    # Return with similarity logs if debugging
    if debug_similarity and hasattr(decoder, "similarity_logs") and decoder.similarity_logs:
        return caption, generated_ids, decoder.similarity_logs[-1]
    else:
        return caption, generated_ids, None 




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
    caption, generated_ids, similarity_logs = generate_caption_3(
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
