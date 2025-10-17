import torch 
from dotenv import load_dotenv

load_dotenv()


def freeze_model_layers(image_encoder, gpt_decoder, freeze_ratio=0.7):

    # Freeze CLIP completely
    for param in image_encoder.parameters():
        param.requires_grad = False


    # Unfreeze last 2 CLIP layers 
    # for layer in image_encoder.visual.transformer.resblocks[-2:]:
    #     for param in layer.parameters():
    #         param.requires_grad = True


    # Freeze embeddings
    for p in gpt_decoder.transformer.wte.parameters():
        p.requires_grad = False
    for p in gpt_decoder.transformer.wpe.parameters():
        p.requires_grad = False

    # for p in gpt_decoder.model.embed_tokens.parameters():
    #         p.requires_grad = False


    # Freeze lower 70% Falcon layers
    # num_layers = len(gpt_decoder.transformer.h)
    # freeze_until = int(0.7 * num_layers)
    # for block in gpt_decoder.transformer.h[:freeze_until]:
    #     for p in block.parameters():
    #         p.requires_grad = False


    # num_layers = len(gpt_decoder.model.layers)
    # freeze_until = int(0.7 * num_layers)


    # Get number of transformer layers
    num_layers = len(gpt_decoder.transformer.h)
    freeze_until = int(freeze_ratio * num_layers)

    # Freeze lower layers
    for block in gpt_decoder.transformer.h[:freeze_until]:
        for p in block.parameters():
            p.requires_grad = False

    # Unfreeze top layers
    for block in gpt_decoder.transformer.h[freeze_until:]:
        for p in block.parameters():
            p.requires_grad = True


    # for block in gpt_decoder.model.layers[:freeze_until]:
    #     for p in block.parameters():
    #         p.requires_grad = False


    # # Unfreeze top 30%
    # for block in gpt_decoder.model.layers[freeze_until:]:
    #     for p in block.parameters():
    #         p.requires_grad = True


    # Trainable: last 30%
    # for block in gpt_decoder.transformer.h[freeze_until:]:
    #     for p in block.parameters():
    #         p.requires_grad = True

            

    # Final LN + LM head
    # for p in gpt_decoder.transformer.ln_f.parameters():
    #     p.requires_grad = True
    # for p in gpt_decoder.lm_head.parameters():
    #     p.requires_grad = True

    # Always unfreeze final norm + lm head
    for p in gpt_decoder.transformer.ln_f.parameters():
        p.requires_grad = True
    for p in gpt_decoder.lm_head.parameters():
        p.requires_grad = True


    # Final norm + lm head
    # for p in gpt_decoder.model.norm.parameters():
    #     p.requires_grad = True
    # for p in gpt_decoder.lm_head.parameters():
    #     p.requires_grad = True

    return image_encoder, gpt_decoder



def calculate_total_train_params(image_encoder, caption_encoder):
    all_params = list([param for param in image_encoder.parameters() if param.requires_grad]) 
    all_params +=  list([param for param in caption_encoder.parameters() if param.requires_grad]) 
    return all_params


def save_to_checkpoint(encoder, decoder, optimizer, epoch, loss, global_step, tmp_suffix=".tmp"):
    print (f" SAVING INTO CHECKPOINT at global step {global_step} and loss {loss}")
    CHECKPOINT_PATH = "checkpoint.pth"
    checkpoint = {
    'encoder_state_dict':encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'gpt_decoder_state_dict': decoder.gpt_decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'global_step': global_step,
    'config': {
        'embed_size': decoder.embed_size,
        'vocab_size': decoder.vocab_size,
        'num_img_tokens': decoder.num_img_tokens
    }
}
    
    tmp_path = CHECKPOINT_PATH + tmp_suffix
    # 1. Save to temporary file
    torch.save(checkpoint, tmp_path)
    import os 
    os.replace(tmp_path, CHECKPOINT_PATH)   

    



def setup_data(N, val_split=0.2):
    from torch.utils.data import Subset
    from torchvision.datasets import CocoCaptions, CocoDetection
    from torchvision import transforms
    from torch.utils.data import random_split


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #              std=[0.229, 0.224, 0.225])
        # transforms.ToTensor()
    ])


    train_dataset_cocooptions = CocoCaptions(
        root='train2017',
        annFile='annotations/captions_train2017.json',
        transform=transform
    )


    train_dataset_detection = CocoDetection(
        root='train2017',
        annFile='annotations/instances_train2017.json',
        transform=transform
    )

    train_dataset_cocooptions = Subset(train_dataset_cocooptions, range(N))
    train_dataset_detection = Subset(train_dataset_detection, range(N))

    # split into train and validation
    val_size = int(len(train_dataset_cocooptions) * val_split)
    train_size = len(train_dataset_cocooptions) - val_size
    train_dataset_cocooptions, val_dataset_cocooptions = random_split(train_dataset_cocooptions, [train_size, val_size])


    val_size = int(len(train_dataset_detection) * val_split)
    train_size = len(train_dataset_detection) - val_size
    train_dataset_detection, val_dataset_detection = random_split(train_dataset_detection, [train_size, val_size])

    return train_dataset_cocooptions, val_dataset_cocooptions, train_dataset_detection , val_dataset_detection




def load_from_checkpoint():

    from decoder_model import ResnetGPT2Wrapper 
    from transformers import GPTNeoForCausalLM, AutoTokenizer
    from encoder import CLIPEncoder

    CHECKPOINT_PATH = "checkpoint.pth"

    checkpoint = torch.load(CHECKPOINT_PATH)

    device = globals().get("device") or "mps"
   
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    global_step = checkpoint["global_step"]


    #### GPT DECODER  ######
    decoder_model_name = "GPT-NEO-350M"


    gpt_decoder = GPTNeoForCausalLM.from_pretrained(decoder_model_name)
    tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    special_tokens = {"additional_special_tokens": ["<START>", "<END>"]}
    tokenizer.add_special_tokens(special_tokens)
    gpt_decoder = GPTNeoForCausalLM.from_pretrained(decoder_model_name)
    gpt_decoder.config.pad_token_id = gpt_decoder.config.eos_token_id
    gpt_decoder.resize_token_embeddings(gpt_decoder.get_input_embeddings().num_embeddings + 2) 

    gpt_decoder.load_state_dict(checkpoint['gpt_decoder_state_dict'])


    ###ENCODER MODEL #####
    encoder = CLIPEncoder(checkpoint['config']['embed_size'])
    encoder = encoder.to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])


    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # Recreate decoder with same config
    decoder = ResnetGPT2Wrapper(
    gpt_decoder=gpt_decoder,
    embed_size=checkpoint['config']['embed_size'],
    vocab_size=checkpoint['config']['vocab_size'],
    num_img_tokens=checkpoint['config']['num_img_tokens'],
    pad_token_id=pad_token_id
           )
    
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device).to(torch.bfloat16)
    decoder = decoder.to(device).to(torch.bfloat16)
    return encoder, decoder , epoch, loss , global_step, tokenizer

