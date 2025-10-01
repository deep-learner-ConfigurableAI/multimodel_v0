import torch.nn as nn 
import torch 

### Decoder Block ####

class ResnetGPT2Wrapper(nn.Module):
    def __init__(self, gpt_decoder, embed_size, vocab_size, pad_token_id, num_heads=4, num_img_tokens=32):
        super().__init__()
        self.gpt_decoder = gpt_decoder
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_img_tokens = num_img_tokens
        self.pad_token_id = pad_token_id

        self.mha = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.key_proj = nn.Linear(embed_size, embed_size)
        self.value_proj = nn.Linear(embed_size, embed_size)
        self.query_proj = nn.Linear(embed_size, embed_size)
        self.layernorm = nn.LayerNorm(embed_size, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_size, eps=1e-6)

        self.dropout = nn.Dropout(0.1)
        self.img_queries = nn.Parameter(torch.randn(num_img_tokens, embed_size) * 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 3 * embed_size),
            nn.GELU(),
            nn.Linear(3 * embed_size, embed_size),
            nn.Dropout(0.1)
        )

        self.to("mps")

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"


    def perform_mha_on_cpu(self, queries, k, v): ##MPS Backend is buggy for MHA 
        self.query_proj = self.query_proj.to("mps")
        self.mha = self.mha.to("mps")
        self.query_proj = self.query_proj.to("mps")
        enriched, _ = self.mha(self.query_proj(queries.to("mps")).to("mps"), k.to("mps"), v.to("mps"))  # (B, M, D)
        enriched = enriched.to("mps")
        self.query_proj = self.query_proj.to("mps")
        return enriched 


    def forward(self, img_features, captions_tensor, attention_mask=None, mode="train"):
        img_features = img_features
         # 1. Token embeddings from GPT2



        #tok_embeds = self.gpt_decoder.transformer.wte(captions_tensor)  # (B, T, D)
        tok_embeds = self.gpt_decoder.get_input_embeddings()(captions_tensor)

        img_features = img_features.to(tok_embeds.device)

        B = tok_embeds.shape[0]

        queries = self.img_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_img_tokens, D)


        # print ("SHAPES")
        # print (f"img_features\t {img_features.shape}")
        # print (f"tok_embeds\t {tok_embeds.shape}")
        # print (f"queries\t {queries.shape}")

        # print ("DEViCES")
        # print (f"img_features device", img_features.device)
        # print (f"tok_embeds", tok_embeds.device)
        # print (f"queries device {queries.device}")
        



        B, T, D = tok_embeds.shape
        N = img_features.shape[1]

        k = self.key_proj(img_features)              # (B, N, D)
        v = self.value_proj(img_features)            # (B, N, D)

        k = k.to(tok_embeds.device)
        v = v.to(tok_embeds.device)

        # print ("k device", k.device)
        # print ("v device", v.device)
        


        # print (f"ATTENTION KEY {k.shape}")
        # print (f"ATTENTION VALUE {v.shape}")

        # print ("DTYPES")
        # print("queries dtype:", queries.dtype)
        # print("k dtype:", k.dtype)
        # print("v dtype:", v.dtype)

        enriched = self.perform_mha_on_cpu(queries, k, v)

        enriched = self.layernorm(queries + enriched) 


        # --- MLP Block ---
        mlp_out = self.mlp(enriched)
        enriched = self.layernorm2(enriched + mlp_out)  #

        fused = torch.cat([enriched, tok_embeds], dim=1)  # (B, M+T, D)


        enriched = self.dropout(fused)


        # enriched = enriched + tok_embeds  # residual connection

        inputs_embeds = enriched[:, :, :].contiguous()


        if attention_mask is not None:
            img_mask = torch.ones(B, self.num_img_tokens, device=tok_embeds.device)
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)
            attention_mask = attention_mask[:, :].contiguous()
            attention_mask = attention_mask.to(tok_embeds.device)
            
            # attention_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)  # (B, 1, 1, seq_len)


        labels = None 
        if mode=='train':
            labels = captions_tensor[:, :].contiguous()
            pad_for_img = torch.full((B, self.num_img_tokens, ), self.pad_token_id, dtype=torch.long, device=tok_embeds.device)
            labels = torch.cat([pad_for_img, labels], dim=1)   # (B, M + T - 1)



        outputs = self.gpt_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )


        # outputs = self.gpt_decoder(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     img_feats=img_features, 
        #     labels=labels
        # )

        return outputs.logits, outputs.loss 
        #return outputs["logits"], outputs["loss"]
