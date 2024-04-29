"""
pytorch's quickstart tutorial

https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

* DataLoader - iterable for loading a Dataset


"""

from typing import Any

import pytest
import torch
from torch import nn
import torch.utils.data as data
import torchvision


pytestmark = pytest.mark.ml


def training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_train_test_data() -> tuple[data.Dataset, data.Dataset]:
    return (
        torchvision.datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        torchvision.datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),  # omitting leaves the dataset in its original format (Image)
        ),
    )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def model_train(
    dataloader: data.DataLoader, model: NeuralNetwork, loss_fn: Any, optimizer: Any
) -> None:
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(training_device()), y.to(training_device())

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def model_test(dataloader: data.DataLoader, model: NeuralNetwork, loss_fn: Any) -> None:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(training_device()), y.to(training_device())
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def test_tutorial() -> None:
    train, test = load_train_test_data()
    assert all([isinstance(d, data.Dataset) for d in [train, test]])

    train_loader = data.DataLoader(train, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(test, batch_size=64, shuffle=True)

    # TODO(@damon): How to strongly type?
    for X, y in test_loader:
        print(type(X))
        print(type(y))
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NeuralNetwork().to(training_device())  # TODO(@damon): why to(device())?
    print(model)
    print(model.parameters())

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # TODO(@damon): find optimal epochs (early stopping)
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}\n--------------------------------------")
        model_train(train_loader, model, loss_fn, optimizer)
        model_test(test_loader, model, loss_fn)
