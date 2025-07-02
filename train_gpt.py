import torch
from torch.nn import functional as F

from model import GPT
from tokenizers import BytePairEncoding
from dataset import get_batch
from configs import *

with open('datasets/tinyshakespeare.txt', 'r') as f:
    dataset = f.read()

train_dataset = dataset[:int(len(dataset) * TRAIN_TEST_SPLIT)]
test_dataset = dataset[int(len(dataset) * TRAIN_TEST_SPLIT):]


tokenizer = BytePairEncoding()
tokenizer.load('sample_byte_pair_encoder')


train_dataset = torch.tensor(tokenizer.tokenize(train_dataset))


model = GPT(vocab_size=tokenizer.vocab_size, count_blocks=3)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)

for step in range(50000):
    x, y = get_batch(train_dataset)
    logits = model(x)
    logits = logits.view(-1, tokenizer.vocab_size)

    y = y.view(-1)
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f'epoch: {step} \t loss: {loss.item()}')

    if step % 300 == 0:
        sample_start = "I'm ok whats up Im looking for my love"
        tokens = torch.tensor(tokenizer.tokenize(sample_start), dtype=torch.long).reshape(1, -1)
        print(tokenizer.detokenize(model.generate_sequence(tokens, 20)[0]))

print(loss.item())
