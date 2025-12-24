import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, d_head, d_model, h_q, h_k): # h_q must be divisible by h_k
        super().__init__()
        self.h_q = h_q
        self.h_k = h_k
        self.d_head = d_head
        self.wq = nn.Linear(d_model, d_head * h_q, bias=False)
        self.wk = nn.Linear(d_model, d_head * h_k, bias=False)
        self.wv = nn.Linear(d_model, d_head * h_k, bias=False)
        self.wo = nn.Linear(d_head * h_q, d_model, bias=False)

    def forward(self, x, mask=True):
        B, T, _ = x.shape
        Q = self.wq(x) # (B, T, d_head * h_q)
        K = self.wk(x) # (B, T, d_head * h_k)
        V = self.wv(x) # (B, T, d_head * h_k)

        Q = Q.reshape(B, T, self.h_q, self.d_head).permute(0, 2, 1, 3)
        K = K.reshape(B, T, self.h_k, self.d_head).permute(0, 2, 1, 3)
        V = V.reshape(B, T, self.h_k, self.d_head).permute(0, 2, 1, 3) # (B, h_k, T, d_head)

        Q = Q.reshape(B, -1, self.h_k, T, self.d_head)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        attn = (Q @ K.transpose(-2, -1)) * (self.d_head ** -0.5)  # (B, h_q/h_k, h_k, T, T)
        if mask:
            causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)  # (B, h_q/h_k, h_k, T, T)
        out = attn @ V  # (B, h_q/h_k, h_k, T, d_head)
        out = out.reshape(B, self.h_q, T, self.d_head).permute(0, 2, 1, 3).reshape(B, T, -1)
        out = self.wo(out) # (B, T, d_model)
        return out

class Transformer(nn.Module): # (B, T, d_model) -> (B, T, d_model)
    def __init__(self, d_model, d_head, h_q, h_k):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.h_q = h_q
        self.h_k = h_k

        self.gqa = GQA(d_head, d_model, h_q, h_k) 
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        
    def forward(self, x):
        x = self.gqa(x)
        x = self.ffn(x)
        return x

class ModAddModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # config: block_size: int = 256 # max sequence length
        # config: vocab_size: int = 83 # number of unique tokens
        # config: n_layer: int = 4 # number of layers
        # config: h_q: int = 12 # number of query heads
        # config: h_k: int = 3 # number of key heads (must divide h_q)
        # config: d_model: int = 768 # embedding dimension
        # config: d_head: int = 64 # head dimension, should be d_model / h_q

        self.model = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.block_size, config.d_model),
            h = nn.ModuleList([Transformer(config.d_model, config.d_head, config.h_q, config.h_k) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Block size is {self.config.block_size}"

        pos = torch.arange(0, T)
        tok_emb = self.model.wte(idx) # (B, T, d_model)
        pos_emb = self.model.wpe(pos).unsqueeze(0) # (1, T, d_model)
        x = tok_emb + pos_emb # (B, T, d_model)

        for block in self.model.h:
            x = block(x)

        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    
    def clear_kv_cache(self):
        for block in self.model.h:
            block.kv_cache = None