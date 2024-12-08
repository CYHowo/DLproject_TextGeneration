import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class CharDataset(Dataset):

    def __init__(self, config, data):

        chars = sorted(list(set(data)))

        # map characters to integer
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = config.block_size

        # encode the string as tensor of character
        self.data = [self.stoi[ch] for ch in data]

    def get_vocab_size(self):
        return len(self.stoi)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data) - self.block_size


    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        # return the chunk and the shifted version as tensors
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

class TransformerConfig:
    def __init__(self, vocab_size, n_layers, n_heads, embed_dim, block_size, dropout):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.dropout = dropout


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.embed_dim % config.n_heads != 0:
            print("!! Embed_dim is not divisible by n_heads.")
        self.kqv_dim = config.embed_dim // config.n_heads

        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                      .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        batch, seq_len, embed = x.size()
        K, Q, V = self.key(x), self.query(x), self.value(x)
        K = K.view(batch, len, -1, self.kqv_dim).transpose(1, 2)
        Q = Q.view(batch, len, -1, self.kqv_dim).transpose(1, 2)
        V = V.view(batch, len, -1, self.kqv_dim).transpose(1, 2)

        att = (Q @ K.transpose(-2, -1)) / (self.kqv_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :len, :len] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ V
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, embed)
        y = self.resid_dropout(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)
        self.self_attention = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        out = x + self.mlp(self.layer_norm2(x))
        return out


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.block_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, idx):
        batch, seq_len = idx.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(pos)

        x = self.dropout(tok_emb + pos_emb)
        for block in self.layers:
            x = block(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_length, temperature=1.0, top_k=None):
        for _ in range(max_length-1):
            logits = self(idx)
            # print(logits[:, -1, :].shape)
            logits = (logits[:, -1, :] / temperature)
            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx