"""
pytorch's quickstart tutorial

https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

from typing import Any

import pytest
import torch
from torch import nn
import torch.utils.data as data
import torchvision


pytestmark = pytest.mark.ml


def training_device() -> torch.device:
    """The recommended way of writing device agnostic code: use the same device
    everywhere"""
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
    """
    The training loop.
    """
    size = len(dataloader.dataset)
    # Turns on automatic differentiation (default_mode or "grad mode") to enable autograd recording.
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(training_device()), y.to(training_device())

        # Set all model gradients to zero. The optimizer stores gradents from
        # previous steps by default.
        optimizer.zero_grad()

        # Run a forward pass
        pred = model(X)

        # Compute loss and gradients
        #
        # Gradients are calculated using automatic differentiation, which is
        # powered by the computational graph that PyTorch constructs during the
        # forward pass. The computational graph encodes the relationships
        # between the input tensors and output tensors of the operations.
        #
        # When you call .backward() on the loss function, this triggers the
        # computation of gradients for all tensors that are involved in
        # computing the loss. PyTorch traverses the computational graph
        # backward, applying the chain rule of calculus to compute gradients of
        # the loss with respect to the parameters of the network.
        loss = loss_fn(pred, y)
        loss.backward()

        # Update model parameters (weights / biases)
        #
        # Step adjusts model parameters in the direction of the gradient, using
        # `learning_rate` and `momentum`.
        #
        # Some optimizers like Adam and RMSProp adaptively adjust the learning
        # rate for each parameter based on the history of gradients, potentially
        # leading to faster convergence and better generalization.
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def model_test(dataloader: data.DataLoader, model: NeuralNetwork, loss_fn: Any) -> None:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # Set the modeul in "evaluation mode"
    # TODO(@damon): What does this actually do?
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
    """
    The training loop:
        * Calculate loss (forward pass)
        * Calculate local gradients
        * Update model parameters
    """
    # Step 1: create Dataset / DataLoader
    train, test = load_train_test_data()
    assert all([isinstance(d, data.Dataset) for d in [train, test]])

    train_loader = data.DataLoader(train, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(test, batch_size=64, shuffle=True)

    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Step 2: Create the model (a.k.a., nn.Module)
    # nn.Modules, like tensors, also have devices
    model = NeuralNetwork().to(training_device())
    print(model)

    # Step 3: Create the loss and optimizer
    loss_fn = nn.CrossEntropyLoss()

    # lr controls the step size. momentum controls inertia, hyperparameter tune
    # these.
    #
    # lr is typically between 1e-2 and 1e-3
    #
    # Momentum allows the optimizer to overcome local minima. Momentum keeps
    # steps large when previous steps were large.
    #
    # momentum is typically between 0.85 and 0.99
    #
    # Good defaults are lr=0.0001 and momentum=0.95
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.95)

    # Step 4: Run the training loop
    # TODO(@damon): find optimal epochs (early stopping)
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}\n--------------------------------------")
        model_train(train_loader, model, loss_fn, optimizer)
        model_test(test_loader, model, loss_fn)
