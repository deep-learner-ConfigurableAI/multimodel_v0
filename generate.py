import torch
from torch.nn import functional as F
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, RepetitionPenaltyLogitsProcessor
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import matplotlib.pyplot as plt

def generate_caption(
    image_tensor,
    encoder,
    decoder, 
    tokenizer, 
    device,
    max_len=30, 
    use_image=True,
    temperature=0.6,  # Reduced temperature for less randomness
    top_k=40,         # More focused token selection
    top_p=0.9,        # More conservative nucleus sampling
    repetition_penalty=1.8,  # Stronger penalty to avoid repetition
    min_length=5,
    debug_similarity=False  # Enable similarity debugging
):
    """
    Generate caption for an image using the encoder-decoder model
    
    Args:
        image_tensor: Image tensor of shape (C, H, W)
        encoder: Image encoder model
        decoder: GPT decoder model wrapper
        tokenizer: Tokenizer for text generation
        device: Device to run inference on
        max_len: Maximum length of generated caption
        use_image: Whether to use image features or generate without image
        temperature: Temperature for sampling (lower = less random)
        top_k: Number of highest probability tokens to keep
        top_p: Cumulative probability for nucleus sampling
        repetition_penalty: Penalty for repeating tokens
        min_length: Minimum length of caption before allowing END token
        
    Returns:
        str: Generated caption
    """
    encoder.eval()
    decoder.eval()

    # Configure similarity debugging if requested
    if debug_similarity:
        decoder.debug_similarity = True
        decoder.similarity_logs = []  # Reset logs

    # Add batch dimension to image if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)

    # Process image through encoder or create dummy embeddings
    if use_image:
        x_embed = encoder(image_tensor)
    else:
        x_embed = torch.zeros((1, 50, decoder.embed_size), device=device)
    
    # Get special token IDs
    start_id = tokenizer.convert_tokens_to_ids("<START>")
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")
    
    # Start with START token
    generated_ids = torch.tensor([[start_id]], device=device)

    # Setup logits processors for text generation
    warpers_list = LogitsProcessorList([
        TemperatureLogitsWarper(temperature),
        TopKLogitsWarper(top_k),
        TopPLogitsWarper(top_p),
    ])
    
    processors_list = LogitsProcessorList([
        RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
        MinLengthLogitsProcessor(min_length, end_token_id)
    ])

    # Implementation of beam search for more coherent captions
    beam_size = 3  # Number of beams to track
    
    # Initialize with start token
    beam_scores = torch.zeros(1, device=device)
    beam_seqs = torch.tensor([[start_id]], device=device)
    beam_finished = [False]
    
    # Generate caption using beam search
    for step in range(max_len):
        # Expand all current beams
        curr_batch_size = beam_seqs.size(0)
        attn_mask = torch.ones(curr_batch_size, beam_seqs.size(1), dtype=torch.long, device=device)
        
        with torch.no_grad():
            # Run inference

            print ("==", x_embed.shape, beam_seqs.shape, attn_mask.shape)
            LLL
            logits, _ = decoder(x_embed.expand(curr_batch_size, -1, -1), beam_seqs, attn_mask, mode="inference")
            
            # Get logits for next token prediction
            next_logits = logits[:, -1, :]
            
            # Apply logits processors for each beam
            for i in range(curr_batch_size):
                next_logits[i] = processors_list(beam_seqs[i:i+1], next_logits[i:i+1].squeeze(0))
                next_logits[i] = warpers_list(beam_seqs[i:i+1], next_logits[i:i+1].squeeze(0))
            
            # Apply softmax to get probabilities
            vocab_size = next_logits.size(-1)
            probs = F.log_softmax(next_logits, dim=-1)  # Using log probabilities for numerical stability
            
            # Add log probs to beam scores
            next_scores = beam_scores.unsqueeze(1) + probs
            
            # Flatten for top-k selection
            flat_next_scores = next_scores.view(-1)
            
            # Select top-k
            best_scores, best_indices = flat_next_scores.topk(beam_size, largest=True, sorted=True)
            
            # Convert flat indices to beam indices and token indices
            beam_indices = best_indices // vocab_size
            token_indices = best_indices % vocab_size
            
            # Create new beam sequences
            new_seqs = []
            new_scores = []
            new_finished = []
            
            for i, (beam_idx, token_idx) in enumerate(zip(beam_indices, token_indices)):
                # Skip if this beam is already finished
                if beam_finished[beam_idx] and i < len(beam_indices) - 1:
                    continue
                    
                new_seq = torch.cat([beam_seqs[beam_idx], token_indices[i:i+1].unsqueeze(0)], dim=1)
                new_seqs.append(new_seq)
                new_scores.append(best_scores[i])
                
                # Check if sequence is finished
                is_finished = token_idx.item() == end_token_id or step == max_len - 1
                new_finished.append(is_finished)
                
                # If we have enough beams, stop
                if len(new_seqs) == beam_size:
                    break
            
            # Update beam state
            beam_seqs = torch.cat(new_seqs, dim=0)
            beam_scores = torch.tensor(new_scores, device=device)
            beam_finished = new_finished
            
            # If all beams are finished, stop
            if all(beam_finished):
                break
    
    # Select best beam
    best_beam_idx = beam_scores.argmax().item()
    generated_ids = beam_seqs[best_beam_idx:best_beam_idx+1]

    # Decode the generated tokens
    caption = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)
    
    # Return caption along with similarity data if debugging was enabled
    if debug_similarity and hasattr(decoder, 'similarity_logs') and decoder.similarity_logs:
        return caption, decoder.similarity_logs[-1]
    else:
        return caption
    

def visualize_caption_1(image_tensor, caption, gt_caption=None, figsize=(10, 8), similarity_data=None):
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



def visualize_caption(image_tensor, caption, gt_caption=None, figsize=(10, 8), similarity_data=None):
    """
    Visualize an image with its generated caption
    
    Args:
        image_tensor: Image tensor
        caption: Generated caption
        gt_caption: Ground truth caption (optional)
        figsize: Figure size
    """
    # Create figure with appropriate subplots
    if similarity_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Convert tensor to numpy for visualization
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # Take first image if batch
        img = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Normalize for visualization if needed
        if img.max() > 1.0:
            img = img / 255.0
    else:
        img = image_tensor
    
    # Plot image and caption
    ax1.imshow(img)
    ax1.axis('off')
    
    title = f"Generated: {caption}"
    if gt_caption:
        title += f"\nGround truth: {gt_caption}"
    
    ax1.set_title(title, fontsize=12)
    
    # Plot similarity data if provided
    if similarity_data:
        # Get data for visualization
        global_to_query = similarity_data['global_to_query']
        
        # Create heatmap
        im = ax2.imshow(global_to_query.reshape(1, -1), cmap='viridis', aspect='auto')
        ax2.set_title('Global-Query Token Similarity')
        ax2.set_xlabel('Query Token Index')
        fig.colorbar(im, ax=ax2, label='Cosine Similarity')
        
        # Add statistics as text
        stats_text = f"Global-Query Mean: {global_to_query.mean():.4f}\n"
        stats_text += f"Global-Query Max: {global_to_query.max():.4f}\n"
        if 'global_to_img' in similarity_data:
            global_to_img = similarity_data['global_to_img']
            stats_text += f"Global-Image Mean: {global_to_img.mean():.4f}"
        
        ax2.text(0.05, 0.05, stats_text, transform=ax2.transAxes, 
                 fontsize=9, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def batch_generate_captions(image_batch, encoder, decoder, tokenizer, device, **kwargs):
    """
    Generate captions for a batch of images
    
    Args:
        image_batch: Batch of image tensors (B, C, H, W)
        encoder: Image encoder
        decoder: Text decoder
        tokenizer: Tokenizer
        device: Device to run on
        **kwargs: Additional arguments for generate_caption
        
    Returns:
        list: List of generated captions and similarity data if debug_similarity is True
    """
    debug_similarity = kwargs.get('debug_similarity', False)
    results = []
    
    for i in range(image_batch.size(0)):
        result = generate_caption(
            image_batch[i], 
            encoder, 
            decoder, 
            tokenizer, 
            device, 
            **kwargs
        )
        results.append(result)
    
    # If debugging is enabled, results contain (caption, similarity_data) tuples
    if debug_similarity:
        captions = [r[0] for r in results]
        similarity_data = [r[1] for r in results]
        return captions, similarity_data
    else:
        return results
