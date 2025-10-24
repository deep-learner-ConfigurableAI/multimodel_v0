import torch 
from dotenv import load_dotenv

load_dotenv()


device = torch.device("mps") 

from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions, CocoDetection
from torchvision import transforms


def freeze_model_layers(image_encoder, gpt_decoder, freeze_ratio=0.5):

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



def calculate_total_train_params(model):
    all_params = list([param for param in model.parameters() if param.requires_grad]) 
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


class CocoAlignedDataset(Dataset):
    def __init__(self, root, caption_ann, detection_ann, transform=None):
        self.caption_dataset = CocoCaptions(root=root, annFile=caption_ann, transform=transform)
        self.detection_dataset = CocoDetection(root=root, annFile=detection_ann, transform=transform)
        
        # Map image_id -> index for each dataset
        self.cap_id_to_idx = {img_id: idx for idx, img_id in enumerate(self.caption_dataset.ids)}
        self.det_id_to_idx = {img_id: idx for idx, img_id in enumerate(self.detection_dataset.ids)}
        
        # Keep only common image IDs
        #self.common_ids = list(set(self.cap_id_to_idx.keys()) & set(self.det_id_to_idx.keys()))
        self.common_ids = [
                img_id for img_id in set(self.cap_id_to_idx.keys()) & set(self.det_id_to_idx.keys())
                if len(self.detection_dataset.coco.getAnnIds(imgIds=[img_id])) > 0
            ]
    
    def __len__(self):
        return len(self.common_ids)
    
    def __getitem__(self, idx):
        image_id = self.common_ids[idx]
        cap_img, captions = self.caption_dataset[self.cap_id_to_idx[image_id]]
        det_img, targets = self.detection_dataset[self.det_id_to_idx[image_id]]

        # --- Get original width and height from COCO metadata ---
        img_info = self.detection_dataset.coco.loadImgs(image_id)[0]
        orig_w, orig_h = img_info["width"], img_info["height"]

        assert cap_img == det_img  # same image (torchvision reuses same file)
        assert len(targets)!=0
        return cap_img, captions, targets, orig_h, orig_w
    


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

    dataset = CocoAlignedDataset(
        root='train2017',
        caption_ann='annotations/captions_train2017.json',
        detection_ann='annotations/instances_train2017.json',
        transform=transform
        
    )

    subset = torch.utils.data.Subset(dataset, range(N))

    val_size = int(N * val_split)
    train_size = N - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(subset, [train_size, val_size])

    # category maps
    coco_api = dataset.detection_dataset.coco
    cat_ids = coco_api.getCatIds()
    cats = coco_api.loadCats(cat_ids)
    id_to_name = {cat["id"]: cat["name"] for cat in cats}
    name_to_id = {cat["name"]: cat["id"] for cat in cats}

    return train_dataset, val_dataset, id_to_name, name_to_id



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

