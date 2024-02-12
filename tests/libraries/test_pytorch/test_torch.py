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
etc) in multiple sequences.

Deep learning typically require large amounts of data (100k+ samples).

PyTorch
-------

Intuitive and user friendly, similar to numpy. Used by researchers and in
industry.

Torch support tabular data by default, other data types with additional
libraries (torchaudio, torchvision, torchtext).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def test_tensor_creation() -> None:
    arr = [[1, 2, 3], [4, 5, 6]]
    tensor = torch.tensor(arr)

    assert isinstance(tensor, torch.Tensor)

    np_arr = np.array(arr)
    np_tensor = torch.from_numpy(np_arr)

    assert isinstance(np_tensor, torch.Tensor)

    assert torch.equal(tensor, np_tensor)

    # tensor attributes
    assert tensor.size() == np_tensor.size()
    assert tensor.size() == torch.Size([2, 3])
    assert tensor.dtype == torch.int64

    tensor = tensor.cpu()  # ensure the tensor is on the CPU

    # the device - typically 'cpu' or 'cuda'
    assert str(tensor.device) == "cpu"


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

    print(linear_layer.weight)
    print(linear_layer.bias)

    output = linear_layer(input_tensor)
    print(output)


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
    print(input_tensor.dtype)
    output_tensor: torch.Tensor = model(input_tensor)

    assert output_tensor.shape == (1, 5)
    print(model(input_tensor))


def test_activation_functions() -> None:
    """
    Linear layers, also known as identity layers, are completely linear. They
    ouput the input data without applying any transformation. They are primarily
    used in models where the output is expected to be a linear combination of
    input features.

    In reality, relationships in data is not completely linear. Using non-linear
    "activation functions" allow us to learn non-linear relationships in data.
    Most NNs contain non-linear activation functions.

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

    # activation functions are used at the last layer of the network
    model = nn.Sequential(
        nn.Linear(6, 4),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    )


def test_forward_pass() -> None:
    """
    The training loop (epoch) consists of:

    1. Propogating data forward
    2. Comparing output to ground truth
    3. Backpropogation to update weights and biases

    Here, we show how to build simple networks for linear classification and
    regression:

    1. Binary classification (using sigmoid)
    2. Multi-class classification (using softmax)
    3. Regression

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
        # it's input layer
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
    labels = F.one_hot(torch.tensor(1), num_classes=3)  # [0, 1, 0]

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

    labels = F.one_hot(torch.tensor(1), num_classes=3)  # [0, 1, 0]

    loss = nn.CrossEntropyLoss()

    loss_good = loss(scores_good.double(), labels.double())
    loss_better = loss(scores_better.double(), labels.double())
    loss_best = loss(scores_best.double(), labels.double())

    assert loss_good > loss_better and loss_better > loss_best


def test_binary_classfication_cross_entropy_loss() -> None:
    """
    Cross entropy loss
    """
    import math

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


def test_backpropogation() -> None:
    """
    Derivatives (gradient) determine how fast and what direction the loss
    function changes.

    Assuming a negative derivative is good? Moving down as we increase X?

    A null (zero?) derivative means the loss function is at it's minimum.

    Backpropogation steps:

    1. Determine gradients (using loss.backward())
    2. Update weights and biases by adjusting each according to the gradient and
       learning rate. The lower the learning rate, the less adjustment per
       epoch.

    Global vs. local minima:

    * Convex functions have a single global minima.
    * Non convex functions have multiple local minima.

    We use an "optimizer" to attempt to find the global minima of non-convex
    functions. The optimizer will update weights and apply an iteration
    algorithm. The most common optimizer is stochastic gradient descent.

    """
    input_data = torch.from_numpy(
        np.random.uniform(low=-1, high=1, size=(1, 16)).astype("float32")
    )
    target = F.one_hot(torch.tensor([1]), num_classes=2)
    model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4), nn.Linear(4, 2))
    prediction = model(input_data)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prediction.double(), target.double())

    assert model[0].weight.grad is None and model[0].bias.grad is None

    # populates the "grad" attribute of the model's weights and biases (how?)
    loss.backward()
    assert model[0].weight.grad is not None and model[0].bias.grad is not None

    before_layer_0_weights: torch.Tensor = model[0].weight
    before_layer_0_biases: torch.Tensor = model[0].bias

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.step()  # update weights

    after_layer_0_weights: torch.Tensor = model[0].weight
    after_layer_0_bias: torch.Tensor = model[0].bias

    # TODO(@damon): Why are the weights the same beore / after .step() which is
    # supposed to update weights?
    print(before_layer_0_weights)
    print(after_layer_0_weights)
