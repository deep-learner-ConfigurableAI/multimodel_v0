import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from torchvision import transforms
from transformers import AutoTokenizer

# Import our models
from setup_model import setup_clip_encoder, setup_gpt_decoder
from decoder_model import ResnetGPT2Wrapper
from generate import generate_caption, visualize_caption

def main():
    parser = argparse.ArgumentParser(description="Generate captions with similarity debugging")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to model checkpoint")
    parser.add_argument("--save_dir", type=str, default="similarity_debug", help="Directory to save similarity visualizations")
    parser.add_argument("--model_path", type=str, default="GPT-NEO-350M", help="Path to the model")
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    clip_encoder = setup_clip_encoder()
    gpt_decoder, tokenizer = setup_gpt_decoder(model_path=args.model_path)
    
    # Get model dimensions and vocabulary size
    embed_size = gpt_decoder.config.hidden_size
    vocab_size = gpt_decoder.config.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Create model with similarity debugging enabled
    model = ResnetGPT2Wrapper(
        gpt_decoder=gpt_decoder,
        embed_size=embed_size,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        num_heads=8,
        num_img_tokens=32,
        debug_similarity=True  # Enable similarity debugging
    )
    
    # Load checkpoint if available
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint: {args.checkpoint}")
    except FileNotFoundError:
        print(f"Warning: No checkpoint found at {args.checkpoint}, using initialized model")
    
    # Load and preprocess image
    try:
        image = Image.open(args.image).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).to(device)
    
    # Generate caption with similarity debugging
    print("Generating caption with similarity debugging...")
    caption_result = generate_caption(
        img_tensor,
        clip_encoder,
        model,
        tokenizer,
        device,
        debug_similarity=True,
        temperature=0.6,
        top_k=40,
        top_p=0.9
    )
    
    if isinstance(caption_result, tuple):
        caption, similarity_data = caption_result
        
        # Print basic similarity statistics
        global_to_query = similarity_data['global_to_query']
        global_to_img = similarity_data['global_to_img']
        
        print("\n--- Similarity Analysis ---")
        print(f"Global-Query Mean: {global_to_query.mean():.4f}, Max: {global_to_query.max():.4f}")
        print(f"Global-Image Mean: {global_to_img.mean():.4f}, Max: {global_to_img.max():.4f}")
        
        # Save full token similarity visualization
        similarity_path = os.path.join(args.save_dir, "token_similarity_full.png")
        model.plot_attention_heatmap(save_path=similarity_path)
        print(f"Token similarity heatmap saved to: {similarity_path}")
        
        # Generate visualization with image and caption
        plt.figure(figsize=(12, 10))
        
        # Main visualization - save to file
        vis_path = os.path.join(args.save_dir, "caption_with_similarity.png")
        visualize_caption(
            img_tensor, 
            caption,
            similarity_data=similarity_data,
            figsize=(12, 6)
        )
        
        print(f"Generated caption: {caption}")
        
        # Generate discriminative power metric
        std_query_sim = np.std(global_to_query)
        mean_query_sim = np.mean(global_to_query)
        discriminative_power = std_query_sim / mean_query_sim
        
        print(f"Discriminative Power: {discriminative_power:.4f}")
        
        if discriminative_power > 0.5:
            print("The global token appears to be learning useful, discriminative information")
        elif discriminative_power > 0.2:
            print("The global token is learning some discriminative information, but could be improved")
        else:
            print("The global token does not appear to be learning sufficiently discriminative information")
    else:
        caption = caption_result
        print(f"Generated caption: {caption}")
        print("No similarity data available. Make sure debug_similarity is set to True.")

if __name__ == "__main__":
    main()
