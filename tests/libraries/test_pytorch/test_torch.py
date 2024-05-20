"""
Deep Learning
-------------
"Deep learning" is the general concept of using a multi-layered model (i.e.,
neural network) to make predictions on unseen data.

What makes deep learning "deep" and why is it so successful?

Deep learning is a hierarchy of "layers". Each layer adds the ability to
recognize different patterns, making the overall network stronger and more
capable.

Traditional machine learning requires hand crafted feature engineering. Deep
learning allows the network to *detect* relevant features from raw data.

Neural networks were inspired by how the brain operates. Neurons operate in
layers, taking the pixels you see and turning them into meaning (flowers, trees,
etc) in multiple sequences. This is why deep learning is often referred to as
"neural networks".

Deep learning typically require large amounts of data (100k+ samples).

PyTorch
-------

Intuitive and user friendly. Used by researchers and in industry.

Torch support tabular data by default, other data types with additional
libraries (torchaudio, torchvision, torchtext).
"""

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def test_tensor_creation() -> None:
    """
    A tensor is the foundational data structure in torch. Tensors are
    multi-dimensional, similar to numpy.array. Tensors can be created in
    multiple ways.
    """
    # creating: from python arrays
    arr = [[1, 2, 3], [4, 5, 6]]
    tensor = torch.tensor(arr, device=torch.device("cpu"))

    assert tensor.size() == (2, 3)
    assert tensor.shape == tensor.size()  # shape and size are synonyms
    assert tensor.dtype == torch.int64
    # the device is typically cpu, but could be cuda (nvidia) or mps (apple metal)
    assert tensor.device == torch.device("cpu")

    # creating: from numpy arrays.
    #
    # tensor (and torch) is numpy friendly. Data
    # can share the same underlying memory via bridging.
    np_arr = np.array(arr)
    np_tensor = torch.from_numpy(np_arr)
    assert torch.equal(tensor, np_tensor)

    # creating: random
    rand_tensor = torch.rand_like(np_tensor.type(torch.float))
    assert rand_tensor.shape == np_tensor.shape == (2, 3)


def test_tensor_operations() -> None:
    a = torch.tensor([[1, 1], [2, 2]])
    b = torch.tensor([[3, 3], [4, 4]])

    # elementwise operations (requires all tensors have the same shape)
    assert torch.equal(torch.tensor([[4, 4], [6, 6]]), a + b)
    assert torch.equal(torch.tensor([[3, 3], [8, 8]]), a * b)


def test_nn_creation() -> None:
    """
    Networks with only linear layers are called "fully connected" networks, as
    all nodes in each adjacent layer are connected.

    A linear layer solves the following formula

    * W = weight matrix
    * X = feature matrix
    * b = bias

    y(0) = W(0) * X + b(0)
    """
    input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])

    linear_layer = nn.Linear(in_features=3, out_features=2)

    # manually perform calculation
    y_manual = torch.matmul(input_tensor, linear_layer.weight.T) + linear_layer.bias

    # use torch to perform calculation
    y_network = linear_layer(input_tensor)

    assert torch.allclose(y_manual, y_network)


def test_sequential() -> None:
    """
    Builds a fully connected neural network with 3 layers.
    """
    model = nn.Sequential(
        nn.Linear(10, 18),
        nn.Linear(18, 20),
        nn.Linear(20, 5),
    )

    input_tensor = torch.from_numpy(
        np.random.uniform(low=-1, high=1, size=(1, 10)).astype("float32")
    )

    output_tensor: torch.Tensor = model(input_tensor)
    assert output_tensor.shape == (1, 5)


def test_activation_functions() -> None:
    """
    Linear layers, also known as identity layers, are completely linear. They
    ouput the input data without applying any transformation. They are primarily
    used in models where the output is expected to be a linear combination of
    input features.

    In reality, relationships in data is not completely linear. Using non-linear
    "activation functions" allow us to learn non-linear relationships in data.
    Most NNs contain non-linear activation functions as the last step (except
    for linear regression).

    Sigmoid is commonly used for binary classification. We send the network
    output to the sigmoid function. If > 0.5, we classify as positive, otherwise
    negative.

    Sigmoid as the last step in a network of linear layers is eqivalent to
    traditional logistic regression.

    Softmax is used for multi-class classification. Each element in the output
    tensor is a probability and all probabilities sum to 1.
    """

    input_tensor = torch.tensor([4.3, 6.1, 2.3])
    probabilities = nn.Softmax(dim=-1)
    output_tensor = probabilities(input_tensor)

    # because [1] has the highest value in the input tensor, it will have the
    # highest softmax value.
    assert torch.argmax(output_tensor) == 1

    # activation functions are used at the last layer of the network (or between
    # linear layers)
    _ = nn.Sequential(
        nn.Linear(6, 4),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    )


def test_relu() -> None:
    """
    Softmax (multi-class classification) and Sigmoid (binary classification) are
    used as the last layer of the network. But activation functions can be used
    *between* layers.

    The gradient of the softmax or tanh function approach zero for high and low
    values of X, leading to low or stalled learning during the training process.
    This is called the "vanishing gradient" problem.

    ReLU will prevent vanishing gradients for positive input values. ReLU can
    suffer from the "dying ReLU" problem, where neurons can get stuck at a state
    of zero activation and never recover during training. If the neuron always
    outputs zero for all values, it will get stuck. LeakyReLU multiplies
    negative values by a small value (default = 0.01) to prevent "dying ReLU".
    It never sets the gradients to zero, allowing every neuron to keep learning.

    ReLU is a good default choice for activation between linear layers.

    Question: why would you *not* always use LeakyReLU?
    """

    relu = nn.ReLU()

    x = torch.tensor(-1.0, requires_grad=True)
    y: torch.Tensor = relu(x)
    assert y.item() == 0.0

    y = relu(torch.tensor(3.0, requires_grad=True))
    assert y.item() == 3.0


##
## Network operations (training and inference)
##


def test_forward_pass() -> None:
    """
    A forward pass forwards data thru the network to the output layer, which
    results in model "predictions".

    Backpropagation is the process of updating weights / biases during model
    training.

    The training loop (epoch) consists of:

    1. Propogating data forward
    2. Comparing output to ground truth (i.e., determining loss)
    3. Backpropogation to update weights and biases

    Here, we show how to build simple networks for linear classification and
    regression:

    1. Binary classification (using sigmoid)
    2. Multi-class classification (using softmax)
    3. Regression (no activation function - regression is completely linear)

    A neural network with a single linear layer followed by a sigmoid function
    is the same as logistic regression.
    """
    input_data = torch.from_numpy(
        np.random.uniform(low=-1, high=1, size=(5, 6)).astype("float32")
    )

    #
    # binary classification model
    #
    model = nn.Sequential(
        nn.Linear(6, 4),
        nn.Linear(4, 1),
        nn.Sigmoid(),  # output bounded between 0 and 1
    )

    output: torch.Tensor = model(input_data)
    assert isinstance(output, torch.Tensor)
    assert output.size() == (5, 1)

    # multi-class classification
    n_classes = 3
    model = nn.Sequential(
        nn.Linear(6, 4),
        nn.Linear(4, n_classes),
        # dim=-1 indicates the last dimension has the same last dimension as
        # it's input layer (n_classes)
        #
        # softmax converts raw scores into probabilities that sum to 1.
        nn.Softmax(dim=-1),
    )
    output: torch.Tensor = model(input_data)

    assert output.size() == (5, n_classes)
    assert torch.allclose(output.sum(dim=1), torch.ones(5))

    # regression - the final layer must return 1 feature (the prediction)
    #
    # In linear regression, where the goal is to predict a continuous value,
    # there is no need for activation functions. The output is a linear
    # combination of the input features, adding an activation function would
    # introduce non-linearity, which is unnecessary.
    model = nn.Sequential(
        nn.Linear(6, 4),
        nn.Linear(4, 1),
    )
    output: torch.Tensor = model(input_data)
    assert output.size() == (5, 1)


def test_cross_entropy_loss() -> None:
    """
    How do we determine model performance (loss) on multi-class classification?

    For each example, we get back a tensor of predictions - one probability for
    each class. We also know the ground truth of the sample, the actual class it
    belongs to.

    We one-hot encode the truth value into a tensor to compare with predictions.

    1. Assuming the actual class is `1`, we one-hot encode the truth value into
       a tensor: [0, 1, 0]
    2. Determine loss by comparing OHE truth to output.

    * Truth (y)           = [0, 1, 0]
    * Predictions (y_hat) = [0.21, 0.77, 0.02]

    Loss == 0.23 ust using straight subtraction to determine loss)

    With binary classification, we typically use CrossEntropyLoss.
    """
    labels = F.one_hot(torch.tensor(1), num_classes=3).type(torch.float)  # [0, 1.0, 0]

    # CrossEntropyLoss (log loss) is the most common loss function for
    # classification problems.
    #
    # Cross-entropy loss penalizes the model more heavily when it confidently
    # predicts the wrong class.
    #
    # The goal of the cross-entropy loss function (or any loss function for that
    # matter) is to minimize the difference between the predicted probability
    # distribution and the true probability distribution, rewarding the model
    # for correct predictions and penalizing it for wrong predictions.

    scores_good = torch.tensor([0.21, 0.77, 0.02])
    scores_better = torch.tensor([0.11, 0.87, 0.02])
    scores_best = torch.tensor([0.01, 0.97, 0.02])

    loss = nn.CrossEntropyLoss()

    loss_good = loss(scores_good, labels)
    loss_better = loss(scores_better, labels)
    loss_best = loss(scores_best, labels)
    assert loss_good > loss_better and loss_better > loss_best


def test_binary_classfication_cross_entropy_loss() -> None:
    """
    Cross entropy loss.

    NOTE: This is *not* how pytorch implements cross entropy loss.
    """

    # assume model predictions returned the following probabilities
    _ = torch.tensor([0.23, 0.77])

    # Cross-entropy loss for binary classification:
    #
    # L(y, y_hat) = -(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    # L(y, y_hat) = -(1 * log(0.77) + (1 - 1) * log(1 - 0.77))
    # L(y, y_hat) = -(1 * log(0.77) + (1 - 1) * log(1 - 0.77))
    # L(y, y_hat) = -(1 * -0.2613647641344075 + (1 - 1) * -1.4696759700589417)
    # L(y, y_hat) = -(1 * -0.2613647641344075 + 0)
    # L(y, y_hat) = -(-0.2613647641344075)
    # L(y, y_hat) = 0.2613647641344075

    y = 1
    y_hat = 0.77
    loss = -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

    assert math.isclose(loss, 0.2613647641344075)

    y = F.one_hot(torch.tensor(1), num_classes=2).type(torch.float)
    print(F.cross_entropy(torch.tensor([0.23, 0.77]), y))


def test_backpropogation() -> None:
    """
    Derivatives (gradients) determine how fast and what direction the loss
    function changes.

    Assuming a negative derivative is good? Moving down as we increase X?

    A null (zero?) derivative means the loss function is at it's minimum.

    Backpropogation steps:
        1. Determine gradients (using loss.backward())
        2. Update weights and biases by adjusting each according to the gradient
           and learning rate. The lower the learning rate, the less adjustment
           per epoch.

    Global vs. local minima:

    * Convex functions have a single global minima.
    * Non convex functions have multiple local minima.

    We use an "optimizer" to attempt to find the global minima of non-convex
    functions. The optimizer will update weights and apply an iteration
    algorithm. The most common optimizer is stochastic gradient descent.
    """

    input_data = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(1, 16))).type(
        torch.float
    )
    target = F.one_hot(torch.tensor([1]), num_classes=2).type(torch.float)

    model = nn.Sequential(
        nn.Linear(16, 8),
        nn.Linear(8, 4),
        nn.Linear(4, 2),
    )
    prediction: torch.Tensor = model(input_data)

    # calcluate loss and compute gradient
    criterion = nn.CrossEntropyLoss()
    loss: torch.Tensor = criterion(prediction, target)

    layer_0: nn.Linear = model[0]
    assert layer_0.weight.grad is None and layer_0.bias.grad is None

    # compute gradients
    #
    # populates the "grad" attribute of the model's weights and biases (how?)
    loss.backward()
    assert layer_0.weight.grad is not None and layer_0.bias.grad is not None

    before_layer_0_weights: torch.Tensor = layer_0.weight.clone()
    before_layer_0_biases: torch.Tensor = layer_0.bias.clone()

    # update model parameters (backpropagation)
    #
    # weights = weights - lr * weights.grad
    # biases = biases - lr * biases.grad
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.step()  # update weights

    after_layer_0_weights: torch.Tensor = layer_0.weight
    after_layer_0_biases: torch.Tensor = layer_0.bias

    assert not before_layer_0_weights.allclose(after_layer_0_weights)
    assert not before_layer_0_biases.allclose(after_layer_0_biases)


def test_regression_loss() -> None:
    """
    L1 loss (Mean Absolute Error)
        * MAE = Mean Absolute Error
        * More robust to outliers as the difference is not squared

    L2 loss (Mean Squared Error)
        * MSE = Mean Squared Error
        * Penalizes large errors more heavily due to squaring, making it
          sensitive to outliers.

    Why use L2?

    Sparcity: L1 loss produces sparse solutions because it encourages some
    weights to become exactly zero, leading to feature selection. L2 tends to
    produce small non-zero weights, but rarely zero.


    * L2 loss (MSE)
    * L1 loss (MAE)
    """
    y_hat = np.array(10)
    y = np.array(1)

    mse_numpy = np.mean((y_hat - y) ** 2)
    print(mse_numpy)

    y_hat_t = torch.tensor(y_hat, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)

    criterion = nn.MSELoss()
    mse_pytorch: torch.Tensor = criterion(y_hat_t, y_t)

    assert mse_numpy == mse_pytorch.item() == 81.0


def test_model_architecture() -> None:
    """
    How do you determine an optimal network architecture?

    The first (input) and last layer (output) are fixed.

    Model capacity is the number of parameters in the model. The more layers,
    the more capacity and can work with more complex data sets.

    A bigger model will take longer to train and could potentially overfit.
    Network architecture is like a hyperparameter, we try multiple combinations
    of depth, breadth, and complexity to determine the optimal size.
    """

    # Parameters
    # --------------------------
    # Layer 1: (8 + 1) * 4 == 36
    # Layer 2: (4 + 1) * 2 == 10
    # Total == 46 parameters

    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.Linear(4, 2),
    )
    total = 0
    for param in model.parameters():
        total += param.numel()

    assert total == 46


def test_learning_rate_and_momentum() -> None:
    """
    The optimizer can dramatically impact training. lr controls the step size.
    momentum controls inertia, hyperparameter tune these.

    lr is typically between 1e-2 and 1e-3

    Momentum allows the optimizer to overcome local minima. Momentum keeps steps
    large when previous steps were large.

    momentum is typically between 0.85 and 0.99

    Good defaults are lr=0.0001 and momentum=0.95
    """
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    inputs = torch.rand((10, 8), dtype=torch.float)
    # random labels
    raw_labels = torch.cat((torch.ones(5), torch.zeros(5))).type(torch.int)
    labels = torch.eye(2)[raw_labels]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.95)

    num_epochs = 10
    previous_loss = sys.float_info.max
    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # compute gradients
        optimizer.step()

        # loss *should * be decreasing but it's not guaranteed
        if loss > previous_loss:
            print("warning: loss is increasing")
        previous_loss = loss


def test_layer_initialization() -> None:
    """
    How do you initialize layers?

    Layer weights by default are generally initialized to small values. Having

    nn.init provides functions to initialize weights to a distribution.

    Transfer learning
    -----------------
    Using a pre-trained model as the starting point for a similar task.

    Use torch.save() and torch.load() to save / load a layer or model
    respectively.

    Fine tuning
    -----------
    A type of tranfer learning. Some layers are frozen (typically the early
    layers) and others are tuned.

    Transfer learning is great for vision and NLP which have high dimensions
    (and require a lot of training time).

    Why small, randomizeed initial weights?
    --------------------------
    * Aviods saturation. Gradients for tanh and sigmoid vanish for large values.
    * Symmerty breaking. Randomness breaks symmetry between neurons in the same
      layer (avoids redundancy as multiple neurons would compute the same
      gradient).
    * Gradient propagation. Large gradients can lead to unstable training
      (exploding gradients). Small gradients lead to slow learning (vanishing
      gradients).
    * Avoiding dead neurons. Saturated or vanished gradients are stuck and
      cannot learn.
    """

    layer = nn.Linear(5, 1)
    # initialize layer weights to the uniform distribution between 0.0 and 1.0
    nn.init.uniform_(layer.weight)
    nn.init.uniform_(layer.bias)

    # Example of freezing a layer, allowing only *some* layers to be tuned.
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.Linear(128, 10),
    )

    # To freeze a layer, remove it from autograd.
    # TODO(@damon): Is there a better way to determine
    for name, param in model.named_parameters():
        if name in ["O.weight", "0.bias"]:
            param.requires_grad = False


#
# Model Evaluation
#


def test_data_loading() -> None:
    """
    Dataset / DataLoader
    """

    raw_training = torch.rand((100, 8), dtype=torch.float32)
    raw_targets = torch.tensor(torch.rand(100) >= 0.5).type(torch.float32)

    dataset = TensorDataset(raw_training, raw_targets)

    # dataset can be indexed
    sample = dataset[0]
    sample_training, sample_target = sample
    assert sample_training.size() == (8,)
    assert sample_target.item() == 0.0 or sample_target.item() == 1.0

    # dataset can be iterated
    for batch_inputs, batch_targets in DataLoader(dataset, batch_size=2, shuffle=True):
        print(type(batch_inputs))
        print(type(batch_targets))
