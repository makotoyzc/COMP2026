"""The CNN shared by every step.

    input: C channels, L x L
    conv k x k, circular padding, ReLU   (one per entry in `layers`)
    global average pool
    linear -> 1 logit   (sigmoid gives p(ordered))

Default layers = ((8, 3), (16, 3)): two 3x3 convs, receptive field 5x5,
1,265 weights for C = 1. Circular padding is the lattice's periodic boundary.
Global pooling makes the output independent of where a feature sits, and lets
one network run on any L.
"""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_ch, layers=((8, 3), (16, 3))):
        super().__init__()
        convs, c = [], in_ch
        for out, k in layers:
            convs += [nn.Conv2d(c, out, k, padding=k // 2, padding_mode="circular"), nn.ReLU()]
            c = out
        self.convs = nn.Sequential(*convs)
        self.head = nn.Linear(c, 1)

    def forward(self, x):
        return self.head(self.convs(x).mean((2, 3))).squeeze(1)


def n_weights(net):
    return sum(p.numel() for p in net.parameters())
