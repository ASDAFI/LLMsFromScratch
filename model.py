import torch
from torch import nn, tensor
from torch.nn import functional as F

from layers import DecoderBlock, position_embedding
from configs import *


class GPT(nn.Module):
    def __init__(self, vocab_size: int, count_blocks: int):
        super().__init__()
        self.embedding_lookup = nn.Embedding(vocab_size, EMBEDDING_SIZE)

        positional_embedding = torch.zeros(size=(CONTEXT_SIZE, EMBEDDING_SIZE))
        for pos in range(CONTEXT_SIZE):
            for i in range(EMBEDDING_SIZE):
                positional_embedding[pos][i] = position_embedding(tensor(pos), tensor(i))
        self.register_buffer('positional_embedding', positional_embedding)

        self.blocks = nn.Sequential(*[DecoderBlock() for _ in range(count_blocks)])
        self.ln_final = nn.LayerNorm(EMBEDDING_SIZE)
        self.logits_maker = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def forward(self, x):
        token_emb = self.embedding_lookup(x)
        seq_len = x.size(1)
        pos_emb = self.positional_embedding[:seq_len, :]
        x = token_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.logits_maker(x)

        return logits


    def generate_sequence(self, idx: tensor, sequence_size: int):
        for i in range(sequence_size):
            idx_cond = idx[:, -CONTEXT_SIZE:]
            logits = self(idx_cond)

            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.concat((idx, next_token_idx), dim=1)
        return idx