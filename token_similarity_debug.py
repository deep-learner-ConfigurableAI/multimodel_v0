import torch
import matplotlib.pyplot as plt
from decoder_model import ResnetGPT2Wrapper
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import os

def analyze_token_similarities(model, image_path, caption, encoder):
    """
    Analyze and visualize token similarities for a given image and caption
    
    Args:
        model: The ResnetGPT2Wrapper model
        image_path: Path to the image
        caption: Caption text
        encoder: Image encoder (CLIP) model
    
    Returns:
        Dictionary of computed similarities
    """
    # Set model to eval mode
    model.eval()
    
    # Enable similarity debugging
    model.debug_similarity = True
    model.similarity_logs = []
    
    # Load and process image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path  # Assume PIL image
        
    # Apply transforms and encode image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        img_features = encoder(img_tensor.to(model.gpt_decoder.device))
    
    # Tokenize caption
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Prepare caption tensor
    caption_tokens = tokenizer(caption, return_tensors="pt").input_ids.to(model.gpt_decoder.device)
    attention_mask = torch.ones_like(caption_tokens)

    print(f"Caption tokens: shape {caption_tokens.shape}")
    
    # Forward pass to trigger similarity computation
    with torch.no_grad():
        model(img_features, caption_tokens, attention_mask=attention_mask, mode="inference")
    
    # Visualize similarities
    model.visualize_similarities()
    model.plot_attention_heatmap()
    
    # Return the computed similarities
    return model.similarity_logs[-1] if model.similarity_logs else None

def visualize_image_with_similarities(image_path, similarities, save_path=None):
    """
    Visualize the image alongside the token similarities
    
    Args:
        image_path: Path to the image
        similarities: Dictionary with similarity data
        save_path: Path to save the visualization
    """
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path  # Assume PIL image
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot the image
    ax1.imshow(np.array(image))
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Plot the similarity heatmap
    global_to_query = similarities['global_to_query']
    im = ax2.imshow(global_to_query.reshape(1, -1), cmap='viridis', aspect='auto')
    ax2.set_title('Global Token - Query Tokens Similarity')
    ax2.set_xlabel('Query Token Index')
    fig.colorbar(im, ax=ax2, label='Cosine Similarity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def compare_multi_image_similarities(model, image_paths, encoder, save_dir=None):
    """
    Compare token similarities across multiple images
    
    Args:
        model: The ResnetGPT2Wrapper model
        image_paths: List of paths to images
        encoder: Image encoder (CLIP) model
        save_dir: Directory to save visualizations
    """
    similarities = []
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        # Create dummy caption
        dummy_caption = "a photo of"
        
        # Analyze similarities
        sim = analyze_token_similarities(model, img_path, dummy_caption, encoder)
        similarities.append(sim)
        
        # Save visualization
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"similarity_image_{i}.png")
            visualize_image_with_similarities(img_path, sim, save_path)
    
    return similarities

def plot_similarity_correlation(similarities, save_path=None):
    """
    Plot correlation between different similarity metrics across images
    
    Args:
        similarities: List of similarity dictionaries
        save_path: Path to save the visualization
    """
    global_query_means = [sim['global_to_query'].mean() for sim in similarities]
    global_img_means = [sim['global_to_img'].mean() for sim in similarities]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(global_query_means, global_img_means, alpha=0.7)
    
    # Add best fit line
    z = np.polyfit(global_query_means, global_img_means, 1)
    p = np.poly1d(z)
    plt.plot(global_query_means, p(global_query_means), "r--", alpha=0.7)
    
    plt.xlabel('Global-Query Similarity')
    plt.ylabel('Global-Image Similarity')
    plt.title('Correlation between Similarity Metrics')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(global_query_means, global_img_means)[0, 1]
    plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Correlation plot saved to {save_path}")
    else:
        plt.show()
