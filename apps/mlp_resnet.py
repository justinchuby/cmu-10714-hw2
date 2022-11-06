import os
import sys
import time
from typing import Optional

import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(0)


def ResidualBlock(
    dim: int, hidden_dim: int, norm=nn.BatchNorm1d, drop_prob: float = 0.1
) -> nn.Module:
    ### BEGIN YOUR SOLUTION
    sequential = (
        nn.Linear(dim, hidden_dim)
        | norm(hidden_dim)
        | nn.ReLU()
        | nn.Dropout(drop_prob)
        | nn.Linear(hidden_dim, dim)
        | norm(dim)
    )
    return nn.Residual(sequential) | nn.ReLU()
    ### END YOUR SOLUTION


def MLPResNet(
    dim: int,
    hidden_dim: int = 100,
    num_blocks: int = 3,
    num_classes: int = 10,
    norm=nn.BatchNorm1d,
    drop_prob: float = 0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob)
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(
    dataloader: ndl.data.DataLoader,
    model: nn.Module,
    opt: Optional[ndl.optim.Optimizer] = None,
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()

    all_corrects = []
    all_losses = []
    model.training = opt is not None

    if model.training:
        for i, batch in enumerate(dataloader):
            batch_images, batch_labels = batch[0], batch[1]
            opt.reset_grad()
            out = model(batch_images)
            loss = loss_func(out, batch_labels)
            corrects = np.argmax(out.numpy(), axis=1) == batch_labels.numpy()
            all_corrects.append(corrects)
            loss.backward()
            all_losses.append(loss.numpy())
            opt.step()
    else:
        for i, batch in enumerate(dataloader):
            batch_images, batch_labels = batch[0], batch[1]
            out = model(batch_images)
            loss = loss_func(out, batch_labels)
            corrects = np.argmax(out.numpy(), axis=1) == batch_labels.numpy()
            all_corrects.append(corrects)
            all_losses.append(loss.numpy())

    error_rate = 1 - np.concatenate(all_corrects).mean()
    loss = np.array(all_losses).mean()
    return error_rate, loss
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
