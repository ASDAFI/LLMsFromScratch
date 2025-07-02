import torch
from torch import nn, cos, sin
from torch.nn import functional as F

from configs import *


def position_embedding(pos, i):
    if i % 2 == 0:
        return sin(pos / 1000 ** (i / EMBEDDING_SIZE))
    else:
        return cos(pos / 1000 ** ((i - 1) / EMBEDDING_SIZE))


class AttentionHead(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(EMBEDDING_SIZE, head_size)
        self.key = nn.Linear(EMBEDDING_SIZE, head_size)
        self.value = nn.Linear(EMBEDDING_SIZE, head_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q_k = q @ k.transpose(-2, -1)
        scaled_qk = q_k / (self.head_size ** 0.5)

        weights = F.softmax(scaled_qk, dim=-1)
        out = weights @ v
        return out


class MaskedAttentionHead(AttentionHead):
    def __init__(self, head_size: int):
        super().__init__(head_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q_k = q @ k.transpose(-2, -1)
        scaled_qk = q_k / (self.head_size ** 0.5)

        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.view(1, seq_len, seq_len)
        masked_qk = scaled_qk.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(masked_qk, dim=-1)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, count_heads: int, attention_type="masked"):
        super().__init__()
        assert EMBEDDING_SIZE % count_heads == 0

        head_size = EMBEDDING_SIZE // count_heads

        if attention_type == "masked":
            self.attentions = nn.ModuleList([MaskedAttentionHead(head_size) for _ in range(count_heads)])
        else:
            self.attentions = nn.ModuleList([AttentionHead(head_size) for _ in range(count_heads)])

        self.projection = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)  # Fixed: added projection layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.attentions], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, 4 * EMBEDDING_SIZE),  # Fixed: 4x intermediate size
            nn.LeakyReLU(),
            nn.Linear(4 * EMBEDDING_SIZE, EMBEDDING_SIZE),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_attention = MultiHeadAttention(8)
        self.attention_dropout = nn.Dropout(0.3)
        self.nn = FeedForwardNetwork()
        self.ln = nn.LayerNorm(EMBEDDING_SIZE)

    def forward(self, x):

        x = self.attention_dropout(self.masked_attention(x)) + x
        x = self.ln(x)
        x = self.nn(x) + x

        return x

