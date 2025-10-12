# QVision MultiModel: Efficient Vision-Language Model with Q-Former Architecture

QVision MultiModel is a lightweight and efficient vision-language model that combines the power of CLIP's visual encoder with a GPT-style decoder, bridging the gap between vision and language tasks. The model leverages a Q-Former architecture inspired by state-of-the-art multimodal systems to achieve high-quality image captioning capabilities while maintaining a small footprint.

## Model Description

The QVision MultiModel uses a combination of:
- **CLIP ViT-Base-Patch32** as a frozen visual encoder
- **Custom Q-Former** with cross-attention mechanisms 
- **GPT-Neo** as a language decoder

The model is optimized for image captioning tasks and can be used for a variety of vision-language applications.

### Key Features

- **Efficient Architecture**: Uses frozen CLIP vision encoder and GPT-Neo language model with trainable bridging layers
- **Q-Former Design**: Employs learnable query tokens to extract relevant visual information
- **Small Footprint**: Significantly smaller than most vision-language models while maintaining quality outputs
- **Low Latency**: Fast inference times suitable for real-time applications
- **Easy Integration**: Simple API for image captioning in various applications

### Model Architecture

Our model employs a sophisticated feature flow mechanism to bridge vision and language:

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│             │     │                  │     │                │
│  Image      │     │  Query Tokens    │     │  Text Tokens   │
│  Features   │─────►  + Cross         │─────►  Generation    │
│  (CLIP)     │     │  Attention       │     │  (GPT-Neo)     │
│             │     │                  │     │                │
└─────────────┘     └──────────────────┘     └────────────────┘
```

### Model Parameters

| Component       | Total Parameters | Trainable Parameters | Percentage Trainable |
|-----------------|-----------------|---------------------|---------------------|
| CLIP Encoder    | 152.06M         | 0M                  | 0%                  |
| GPT-Neo Decoder | 355.80M         | 240.36M             | 67.56%              |
| Custom Q-Former | 66.20M          | 66.20M              | 100%                |
| **Total**       | **574.06M**     | **306.56M**         | **53.40%**          |

## SafeTensors Format

This model is provided in the SafeTensors format, which offers several benefits over traditional PyTorch checkpoint files:

1. **Security**: SafeTensors provides better protection against arbitrary code execution attacks that can occur when loading PyTorch pickled files.

2. **Faster Loading**: SafeTensors can be memory-mapped and loaded much faster, especially for large models, reducing startup time.

3. **Language Agnostic**: The format is not tied to Python or PyTorch, making it more portable across different frameworks and programming languages.

4. **Improved Sharing**: Model weights can be shared more confidently without security concerns.

5. **Partial Loading**: Supports efficient loading of specific weights without loading the entire model into memory.

## Usage

To use this model for image captioning:

```python
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def load_model_from_hub(model_name="verma75preetam/qvision-mutlimodel-base"):
    """
    Load the model directly from Hugging Face Hub.
    """
    # Download the model files
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    encoder_path = hf_hub_download(repo_id=model_name, filename="encoder.safetensors")
    decoder_path = hf_hub_download(repo_id=model_name, filename="decoder.safetensors")
    gpt_decoder_path = hf_hub_download(repo_id=model_name, filename="gpt_decoder.safetensors")
    
    # Download the helper scripts
    encoder_py_path = hf_hub_download(repo_id=model_name, filename="encoder.py")
    decoder_model_py_path = hf_hub_download(repo_id=model_name, filename="decoder_model.py")
    generate_py_path = hf_hub_download(repo_id=model_name, filename="generate.py")
    
    # Add the directory containing the scripts to the path
    import sys, os
    sys.path.append(os.path.dirname(encoder_py_path))
    
    # Import the model classes
    from encoder import CLIPEncoder
    from decoder_model import ResnetGPT2Wrapper
    from transformers import GPTNeoForCausalLM, AutoTokenizer
    
    # Load tokenizer
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
    encoder_state_dict = load_file(encoder_path)
    decoder_state_dict = load_file(decoder_path)
    gpt_decoder_state_dict = load_file(gpt_decoder_path)
    
    # Load state dictionaries into models
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    decoder.gpt_decoder.load_state_dict(gpt_decoder_state_dict)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = encoder.to(device).to(torch.bfloat16)
    decoder = decoder.to(device).to(torch.bfloat16)
    
    return encoder, decoder, tokenizer

def generate_caption_for_image(image_path, encoder, decoder, tokenizer):
    """
    Generate a caption for an image using the loaded model.
    """
    # Import the generate function from the downloaded module
    import sys
    from generate import generate_caption
    
    # Set device
    device = next(encoder.parameters()).device
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
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
    
    return caption

# Example usage
encoder, decoder, tokenizer = load_model_from_hub()
image_path = "path/to/your/image.jpg"
caption = generate_caption_for_image(image_path, encoder, decoder, tokenizer)
print(f"Generated caption: {caption}")
```

## Training Data

The model was trained on the MS COCO dataset, which contains:
- 80,000 training images
- Each image has multiple human-annotated captions
- Diverse set of everyday scenes and objects

## Performance

The model achieves strong performance on image captioning tasks with significantly fewer parameters than comparable models:

| Model             | Total Parameters | Trainable Parameters | Inference Time (ms) |
|-------------------|-----------------|---------------------|---------------------|
| QVision MultiModel| 574M            | 307M                | ~50                 |
| BLIP-2            | ~7B             | ~7B                 | ~250                |
| LLaVA             | ~13B            | ~13B                | ~500                |
| CLIP + GPT-4      | ~20B            | ~20B                | ~2000               |

## Limitations

- The model is specialized for image captioning and may not perform as well on other vision-language tasks without fine-tuning
- Maximum caption length is limited to 20 tokens in the default configuration
- Performance may degrade on images that are substantially different from the COCO dataset

## Citation

If you use this model in your research, please cite:

```
@misc{qvision-multimodel,
  author = {Preetam Verma},
  title = {QVision MultiModel: Efficient Vision-Language Model with Q-Former Architecture},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/deep-learner-ConfigurableAI/multimodel_v0}}
}
```

## Example Notebook

We provide a comprehensive Jupyter notebook to help you get started with the model:

```python
# using_model_from_huggingface.ipynb

# This notebook demonstrates:
# 1. Loading the model from Hugging Face Hub
# 2. Processing images for captioning
# 3. Generating captions for single images
# 4. Batch processing multiple images
# 5. Visualizing results
```

The notebook contains step-by-step instructions and code examples for various use cases. It's available in the repository as `using_model_from_huggingface.ipynb`.

## License

This model is released under the MIT License.
