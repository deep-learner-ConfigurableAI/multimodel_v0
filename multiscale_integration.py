import torch
import torch.nn as nn
from multiscale_encoder import MultiScaleCLIPEncoder
from decoder_model import ResnetGPT2Wrapper  # Your existing decoder

def create_multiscale_model(
    embed_size=768,
    model_name="openai/clip-vit-base-patch32",
    decoder_config=None,
    num_img_tokens=64,
    num_heads=8,
):
    """
    Create a complete model with MultiScaleCLIPEncoder and ResnetGPT2Wrapper
    
    Args:
        embed_size: Embedding dimension for the model
        model_name: CLIP model name to use
        decoder_config: Configuration for the decoder (GPT-2 model)
        num_img_tokens: Number of image tokens to use in the decoder
        num_heads: Number of attention heads in the encoder
    
    Returns:
        encoder: MultiScaleCLIPEncoder instance
        decoder: ResnetGPT2Wrapper instance
    """
    # Create multiscale encoder
    encoder = MultiScaleCLIPEncoder(
        embed_size=embed_size,
        model_name=model_name,
        freeze_vision=True,
        num_heads=num_heads,
        num_points=4,
        dropout=0.1
    )
    
    # Create decoder (using your existing configuration)
    if decoder_config is None:
        # Default configuration if none provided
        from transformers import GPT2LMHeadModel
        gpt_decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        vocab_size = gpt_decoder.config.vocab_size
        pad_token_id = 0  # Assuming pad token is 0
    else:
        gpt_decoder = decoder_config["gpt_model"]
        vocab_size = decoder_config["vocab_size"]
        pad_token_id = decoder_config["pad_token_id"]
    
    decoder = ResnetGPT2Wrapper(
        gpt_decoder=gpt_decoder,
        embed_size=embed_size,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        num_heads=num_heads,
        num_img_tokens=num_img_tokens,
        debug_similarity=False
    )
    
    return encoder, decoder

def generate_captions_with_multiscale_encoder(
    image_tensor,
    encoder,
    decoder,
    tokenizer,
    device,
    max_len=30,
    temperature=0.7,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.5
):
    """
    Generate captions using the multiscale encoder and existing decoder
    
    Args:
        image_tensor: Input image tensor [1, 3, H, W]
        encoder: MultiScaleCLIPEncoder instance
        decoder: ResnetGPT2Wrapper instance
        tokenizer: Tokenizer for decoding/encoding text
        device: Device to run on
        max_len: Maximum length of the caption
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p filtering parameter
        repetition_penalty: Penalty for repeating tokens
        
    Returns:
        caption: Generated caption
    """
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
    )
    
    encoder.eval()
    decoder.eval()
    
    # Make sure image has batch dimension
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    # Get multiscale features from the encoder
    with torch.no_grad():
        x_embed = encoder(image_tensor)
    
    # Start with the START token
    start_id = tokenizer.convert_tokens_to_ids("<START>")
    generated_ids = torch.tensor([[start_id]], device=device)
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")
    
    # Setup warpers and processors for better text generation
    warpers_list = LogitsProcessorList([
        TemperatureLogitsWarper(temperature),
        TopKLogitsWarper(top_k),
        TopPLogitsWarper(top_p),
    ])
    
    processors_list = LogitsProcessorList([
        RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
    ])
    
    # Generate caption token by token
    for _ in range(max_len):
        with torch.no_grad():
            # Create attention mask for the current sequence
            attn_mask = torch.ones(1, generated_ids.shape[1], dtype=torch.long, device=device)
            
            # Use the decoder in inference mode
            logits, bbox_preds, objectness_pred, class_pred, _ = decoder(
                x_embed, 
                generated_ids, 
                attn_mask, 
                mode="inference"
            )
            
            # Get logits for the next token
            next_logits = logits[:, -1, :]
            
            # Process logits to avoid repetition
            next_logits = processors_list(generated_ids, next_logits)
            
            # Apply temperature and top-k/top-p filtering
            next_logits = warpers_list(generated_ids, next_logits)
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(next_logits, dim=-1)
            
            # Sample the next token
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Add the new token to our sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Stop if we generated an END token
            if next_token_id.item() == end_token_id:
                break
    
    # Decode the generated tokens to text, skipping special tokens
    caption = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)
    
    # Process detection results
    bboxes = bbox_preds[0].cpu().numpy()  # [num_tokens, 4]
    class_ids = class_pred[0].cpu().numpy()
    scores = objectness_pred[0].cpu().numpy()
    
    # Create detection results
    detections = []
    for (cx, cy, w, h), score, cls_id in zip(bboxes, scores, class_ids):
        if score > 0.2:  # confidence threshold
            detections.append({
                "bbox": [cx, cy, w, h],  # Normalized coordinates
                "score": float(score),
                "class_id": int(cls_id)
            })
    
    return caption, detections

# Example of how to use the multiscale model in your training loop
def example_training_step(batch, encoder, decoder, optimizer, device):
    """Example training step with multiscale encoder"""
    # Unpack batch
    image_tensor = batch["images"].to(device)
    caption_tensor = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    bboxes = batch["bboxes"].to(device)
    class_labels = batch["class_labels"].to(device)
    objectness = batch["objectness"].to(device)
    
    # Forward pass with multiscale encoder
    x_embed = encoder(image_tensor)
    
    # Forward pass with decoder
    logits, bbox_preds, objectness_pred, class_pred, loss = decoder(
        x_embed,
        caption_tensor,
        attention_mask,
        bbox_targets=bboxes,
        class_targets=class_labels,
        objectness_targets=objectness,
        mode="train"
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
