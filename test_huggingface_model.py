"""
Test script to load and use the multimodal model directly from Hugging Face Hub.
"""

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def load_model_from_hub(model_name="verma75preetam/qvision-mutlimodel-base"):
    """
    Load the model directly from Hugging Face Hub.
    """
    print(f"Loading model from Hugging Face Hub: {model_name}")
    
    # Import necessary dependencies
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
    # Download the model files
    print("Downloading model files...")
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    encoder_path = hf_hub_download(repo_id=model_name, filename="encoder.safetensors")
    decoder_path = hf_hub_download(repo_id=model_name, filename="decoder.safetensors")
    gpt_decoder_path = hf_hub_download(repo_id=model_name, filename="gpt_decoder.safetensors")
    
    # Download the helper scripts
    encoder_py_path = hf_hub_download(repo_id=model_name, filename="encoder.py")
    decoder_model_py_path = hf_hub_download(repo_id=model_name, filename="decoder_model.py")
    generate_py_path = hf_hub_download(repo_id=model_name, filename="generate.py")
    
    # Add the directory containing the scripts to the path
    import sys
    sys.path.append(os.path.dirname(encoder_py_path))
    
    # Import the model classes
    from encoder import CLIPEncoder
    from decoder_model import ResnetGPT2Wrapper
    from transformers import GPTNeoForCausalLM, AutoTokenizer
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("GPT-NEO-350M")
    special_tokens = {"additional_special_tokens": ["<START>", "<END>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    # Load configuration
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize models
    print("Initializing model components...")
    encoder = CLIPEncoder(config['embed_size'])
    
    # Initialize GPT model
    gpt_model = GPTNeoForCausalLM.from_pretrained("GPT-NEO-350M")
    gpt_model.resize_token_embeddings(gpt_model.get_input_embeddings().num_embeddings + 2)  # For special tokens
    
    # Initialize decoder
    decoder = ResnetGPT2Wrapper(
        gpt_decoder=gpt_model,
        embed_size=config['embed_size'],
        vocab_size=config['vocab_size'],
        num_img_tokens=config['num_img_tokens'],
        pad_token_id=pad_token_id
    )
    
    # Load weights from safetensors
    print("Loading model weights...")
    encoder_state_dict = load_file(encoder_path)
    decoder_state_dict = load_file(decoder_path)
    gpt_decoder_state_dict = load_file(gpt_decoder_path)
    
    # Load state dictionaries into models
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    decoder.gpt_decoder.load_state_dict(gpt_decoder_state_dict)
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else 
                         "cpu")
    
    print(f"Using device: {device}")
    encoder = encoder.to(device).to(torch.bfloat16)
    decoder = decoder.to(device).to(torch.bfloat16)
    
    print("Model successfully loaded from Hugging Face Hub!")
    return encoder, decoder, tokenizer

def generate_caption_for_image(image_path, encoder, decoder, tokenizer):
    """
    Generate a caption for an image using the loaded model.
    """
    # Import the generate function from the downloaded module
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from generate import generate_caption, visualize_caption
    
    # Set device
    device = next(encoder.parameters()).device
    
    print(f"Loading and processing image: {image_path}")
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    print("Generating caption...")
    # Generate caption
    caption, generated_ids, _ = generate_caption(
        image_tensor,
        encoder,
        decoder,
        tokenizer,
        device,
        temperature=0.7,
        repetition_penalty=1.5
    )
    
    # Display results
    print(f"\nGenerated caption: {caption}")
    
    # Create a simple display for the image and caption
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.5, f"Caption: {caption}", fontsize=12, wrap=True)
    plt.axis("off")
    plt.title("Generated Caption")
    
    plt.tight_layout()
    plt.savefig("huggingface_model_test_result.png")
    plt.show()
    
    return caption

if __name__ == "__main__":
    # Path to a test image
    test_image_path = "train2017/000000000009.jpg"  # Use a sample image from your dataset
    
    # Load model from Hugging Face Hub
    encoder, decoder, tokenizer = load_model_from_hub()
    
    # Generate caption for test image
    caption = generate_caption_for_image(test_image_path, encoder, decoder, tokenizer)
    
    print(f"\nCaption generated from Hugging Face model: {caption}")
    print("\nTest completed successfully!")
