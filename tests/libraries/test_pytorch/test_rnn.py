"""
Recurrent neural networks (RNNs) process sequential data such as time series,
text, or audio.

It's critical when splitting train / test data that we do not split randomly.
For time series, we want to split based on time. For example, if you have 3
years of training data, train on the first two years, using the last year as the
test set.

Each training example should be a full sequence. For example, an hour of data or
a day of data.

What makes a network or neuron recurrent?

Neurons maintain a hidden state. The hidden state is updated with every
prediction and is used in calculating the output. The hidden state maintains
memory through time.


RNN Architectures
-----------------

* Sequence to sequence architectures. Passes the entire sequence as input, use
  the entire output. Example: audio recognition. Receive an entire waveform,
  output text.

* Sequence to vector. Pass sequence as input, use only the last output. Example:
  text topic classification. Read the whole text, then output a single topic.

* Vector to sequence. Pass a single input, use the entire output sequence.
  Example: text generation. Given a sentiment or word, create a sentence from
  it.

* Encoder-decoder architecture. Pass entire input sequence, only then start
  using output sequences. The entire input must be processed before output can
  be generated. Example: translating text from one language to another.



Cell Memory
-----------
An RNN cell has the following properties:

* Two inputs: current input data (x), previous hidden state (h).
* Two outputs: current ouput (y), next hidden state (h).

RNNs have short term memory. They do not carry hidden state between predictions.
There are two more advanced RNNs which maintain longer term state.

* LSTM (Long Short Term Memory)
* GRU (Gated Recurrent Unit)

An LSTM cell has the following properties:

* Three inputs: short term state (h) and long term state (c)

* Three gates
    * Forget gate: what to remove from long-term memory
    * Input gate: what to save to long-term memory
    * Output gate: what to return at the current time step

* Three outputs:
    * y
    * short term hidden state (h)
    * long term hidden state (c)


The GRU cell is similar to an LSTM cell, however it maintains just one hidden
state and doesn't use an output gate.

What type of RNN shoudl we use?

* RNNs aren't used much anymore. We should choose between LSTM / GRU.
* GRU is less computation.
* Try both and compare.




"""

from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

import torchmetrics
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


def test_rnn() -> None:
    """
    Uses a sequence to vector RNN architecture. An entire day's worth of
    predictions are used to predict the output at the beginning of the next day.
    """

    def create_sequences(
        df: pd.DataFrame, seq_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []

        # assumes the target element is the last element in the sequence
        for i in range(len(df) - seq_length):
            x = df.iloc[i : (i + seq_length), 1]
            y = df.iloc[i + seq_length, 1]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    data_train = pd.read_csv("./data/electricity/electricity_train.csv")
    # Readings are taken every 15 minutes. Each training sample is one full day,
    # with the target being the next immediate data element (the start of the
    # next day).
    seq_length = 24 * 4

    X_train, y_train = create_sequences(data_train, seq_length)
    dataset_train = TensorDataset(
        torch.from_numpy(X_train).float().to(training_device()),
        torch.from_numpy(y_train).float().to(training_device()),
    )
    dataloader_train = DataLoader(dataset_train, batch_size=1024, shuffle=False)

    data_test = pd.read_csv("./data/electricity/electricity_test.csv")
    X_test, y_test = create_sequences(data_test, seq_length)
    dataset_test = TensorDataset(
        torch.from_numpy(X_test).float().to(training_device()),
        torch.from_numpy(y_test).float().to(training_device()),
    )
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

    # sequence to vector RNN (we are processing all input, only using the last
    # value as output)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=1,  # we only have 1 feature (energy consumption)
                hidden_size=32,  # arbitrary
                num_layers=2,  # arbitrary
                batch_first=True,  # the data will have the batch size as the first dimension
            )

            # from the hidden size of 32 -> 1 for the final prediction
            self.fc = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor):
            # 2 = num_layers
            # input size (the first dimension (96)
            # hidden state size (initialize to zero)
            h0 = torch.zeros(2, x.size(0), 32).to(training_device())
            out, _ = self.gru(x, h0)
            # The prediction is the last element in the 2nd dimension
            out = self.fc(out[:, -1, :])
            return out

    model = Net().to(training_device())

    # We are doing a regression: therefore use MSELoss.
    #
    # MSELoss has the following properties
    #
    # * Ensures positive and negative errors don't cancel out.
    # * Penalizes large errors more.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # NOTE: there is a vector mismatch. I don't understand what structure the hidden
    for epoch in range(10):
        model.train()
        training_loss: float = 0.0
        for X, y in dataloader_train:
            #
            # All RNNs expect input shape of
            #
            # (batch_size, seq_length, num_features)
            #
            # unsqueeze the 2nd dimension to turn a batch into rows
            X = cast(torch.Tensor, X).view(len(X), seq_length, 1)
            y = cast(torch.Tensor, y)

            optimizer.zero_grad()

            # The output is in shape (batch_size, 1). y is in shape
            # (batch_size).
            #
            # Squeezing the model output will put it into (batch_size) shape.
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        epoch_loss = training_loss / len(dataloader_train)
        print(f"epoch {epoch}: loss {epoch_loss}")

        mse = torchmetrics.MeanSquaredError().to(training_device())
        model.eval()
        with torch.no_grad():
            for X, y in dataloader_test:
                X = cast(torch.Tensor, X).view(len(X), seq_length, 1)
                y = cast(torch.Tensor, y)

                outputs = model(X).squeeze()
                mse(outputs, y)
            print(f"Test MSE: {mse.compute()}")
