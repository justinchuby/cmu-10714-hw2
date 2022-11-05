import os
import sys
import time

import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1) -> nn.Module:
    ### BEGIN YOUR SOLUTION
    sequential = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Residual(sequential)
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob)
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
