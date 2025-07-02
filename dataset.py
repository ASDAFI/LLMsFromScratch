import torch

from configs import *

def get_batch(dataset: torch.Tensor):
    start_indexes = torch.randint(low=0,
                                  high=len(dataset) - CONTEXT_SIZE - 1,

                                  size=(BATCH_SIZE,))
    x = torch.stack([dataset[idx: idx + CONTEXT_SIZE]
                     for idx in start_indexes])
    y = torch.stack([dataset[idx + 1: idx + CONTEXT_SIZE + 1]
                     for idx in start_indexes])
    return x, y
