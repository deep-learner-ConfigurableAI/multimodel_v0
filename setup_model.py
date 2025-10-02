import torch 
from pydantic import BaseModel
from utils import freeze_model_layers, calculate_total_train_params
from typing import ClassVar

from dotenv import load_dotenv

load_dotenv()


device = torch.device("mps")

def get_models():
    from transformers import GPTNeoForCausalLM, AutoTokenizer
    from encoder import CLIPEncoder
    from decoder_model import ResnetGPT2Wrapper

    model_name_or_path = "GPT-NEO-350M"  

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    gpt_model = GPTNeoForCausalLM.from_pretrained(model_name_or_path)

    special_tokens = {"additional_special_tokens": ["<START>", "<END>"]}
    tokenizer.add_special_tokens(special_tokens)

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    start_token_id = tokenizer.convert_tokens_to_ids("<START>") 
    end_token_id = tokenizer.convert_tokens_to_ids("<END>") 

    gpt_model.resize_token_embeddings(gpt_model.get_input_embeddings().num_embeddings + 2)  

    gpt_model = gpt_model.to(device)

    class TrainingConfig(BaseModel):
        gpt_hidden_size: ClassVar[int] = gpt_model.config.hidden_size
        embed_size: ClassVar[int] = gpt_hidden_size      # to match GPT hidden size
        hidden_size: ClassVar[int] = gpt_hidden_size
        batch_size: ClassVar[int] = 4
        input_channels: ClassVar[int] = 3
        image_h: ClassVar[int] = 224
        image_w: ClassVar[int] = 224
        steps: ClassVar[int] = 0
        epochs: ClassVar[int] = 10
        lr: ClassVar[float] = 2e-5
        accumulation_steps: ClassVar[int] = 4
        vocab_size: ClassVar[int] = gpt_model.config.vocab_size
        number_of_items : ClassVar[int] = 80000
        caption_len : ClassVar[int] = 20 

    encoder_model = CLIPEncoder(TrainingConfig.embed_size)
    encoder_model, gpt_model = freeze_model_layers(encoder_model, gpt_model)
    decoder_model = ResnetGPT2Wrapper(gpt_model, TrainingConfig.embed_size, TrainingConfig.vocab_size, pad_token_id)

    all_params = calculate_total_train_params(encoder_model, decoder_model)

    print ("Trainable parameters in  model:")
    print (sum([p.numel() for p in all_params if p.requires_grad]))

    decoder_model = decoder_model.to(device)
    encoder_model = encoder_model.to(device)

    return TrainingConfig, encoder_model, decoder_model , pad_token_id, tokenizer








