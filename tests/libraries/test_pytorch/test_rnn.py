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
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

import torchmetrics
import torchvision


def test_rnn() -> None:
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

    train_data = pd.read_csv("./data/electricity/electricity_train.csv")
    # Readings are taken every 15 minutes. Each training sample is one full day,
    # with the target being the next immediate data element (the start of the
    # next day).
    seq_length = 24 * 4

    X_train, y_train = create_sequences(train_data, seq_length)
    dataset_train = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )

    # sequence to vector RNN (we are processing all input, only using the last
    # value as output)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(
                input_size=1,  # we only have 1 feature (energy consumption)
                hidden_size=32,  # arbitrary
                num_layers=2,  # arbitrary
                batch_first=True,  # the data will have the batch size as the first dimension
            )
            self.fc = nn.Linear(
                32, 1
            )  # from the hidden size of 32 -> 1 for the final prediction

        def forward(self, x: torch.Tensor):
            # 2 = num_layers
            # input size
            # hidden state size
            h0 = torch.zeros(2, x.size(0), 32)
            out, _ = self.rnn(x, h0)
            # The prediction is the last element in the 2nd dimension
            out = self.fc(out[:, -1, :])
            return out
