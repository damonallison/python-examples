"""
Convolutional neural networks perform "convolution" operations on a set of
pixels at a time, which provides the ability for networks to learn patterns
(edges, curves, etc) as well as reduces the overall number of parameters in the
network.

"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

import torchmetrics
import torchvision

pytestmark = pytest.mark.ml


def test_images_cnn() -> None:
    """
    What is an image? A matrix of pixels (picture element). Each pixel is
    typically described in RGB format

    RGB = (51, 171, 214)

    Data Augmentation
    -----------------

    Data augmentation is the process of generating new training data by
    augmenting existing training data. Data augmentation increases model
    robustness by:

    * Increasing training set size.
    * Training on "real world" data. Real-world data is not perfect.
    * Reduces overfitting.

    Model Architecture (Convolutions)
    ------------------

    Why not use linear layers? Images are large. Having a parameter per pixel is
    too expensive. A 256x256x3 image is ~ 200k input parameters. A layer with
    100 neurons would contain 200M parameters. Linear layers also don't
    recognize patterns of spacial related pixels.


    Convolutional layers slide filters (small grid) over the image, performing a
    convolution at each position. This allows us to perserve input patters and
    uses fewer parameters than a linear layer.

    # input feature maps, output feature maps, kernel_size=3

    nn.Conv2d(3, 32, kernel_size=3)

    What is a convolution?

    * The sum of the dot product of the input patch and filter.

    * Zero padding adds a frame of zeros to the convolutional layers input (not
      patch). This maintains spacial dimensions of input and output tensors and
      ensures border pixels are treated equally to others. If we *don't* have
      padding, border pixels wouldn't have as many filters pass over them.

    * Max Pooling is another technique commonly used after convolutional layers.
      It slides a non-overlapping window over the output and takes the max value
      for all elements in the window. This reduces the dimensions and parameters
      in the network.

    nn.MaxPool2d(kernel_size=2)
    """

    train_transformers = torchvision.transforms.Compose(
        [
            # Agument the data by introducing randomness to the image, making
            # the model more robost to real world images. Be careful *not* to
            # augment the data in ways that would change the label. For example,
            # augmenting the color of a lemon could confuse it for a lime.
            # Always consider your dataset before applying transformations.
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.RandomAutocontrast(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((64, 64)),
        ]
    )

    dataset_train = torchvision.datasets.ImageFolder(
        "data/clouds/clouds_train",
        transform=train_transformers,
    )
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    #
    # To display an image, put the image into a shape compatible w/ matplotlib
    #
    # image, label = next(iter(dataloader_train))
    # # Remove the first dimension
    # img_to_show = image.squeeze()
    # # Height,Width,Channel
    # img_to_show = img_to_show.permute(1, 2, 0)
    # plt.imshow(img_to_show)
    # plt.show()

    class ConvNet(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()

            # This CNN has two parts: a feature_extractor and a classifier
            #
            # Most CNN architectures look similar: repeated blocks of
            # convolution / activation / pooling layers with increasing number
            # of feature outputs (why increasing?), followed by flatten and one
            # or more layers for classification or regression.
            self.feature_extractor = nn.Sequential(
                # The input feature map has 3 features corresponding to the RGB channels.
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
            )

            # how do we get 16x16?
            #
            # Input image = 3 x 64 x 64
            #
            # Conv2d(3, 32) == 32 output filters = 32 * 64 * 64
            #
            # MaxPool2d(2) == halve the width / height = 32 * 32 * 32
            #
            # Conv2d(32, 64) == 64 ourput filters = 64 * 32 * 32
            #
            # MaxPool2d(2) == halve the width / height = 64 * 16 * 16

            self.classifier = nn.Linear(
                in_features=64 * 16 * 16,
                out_features=num_classes,
            )

        def forward(self, x):
            x = self.feature_extractor(x)
            x = self.classifier(x)
            return x

    # Data agumentation and how it can impact the training process:
    #
    # Consider image implications when augmenting data! If the augmentation
    # would change the label, do *not* augment along that dimension.
    #
    # Examples:
    #
    # * Color: Augmenting color will confuse the model. lemon / lime are the
    #   same image w/ different colors.
    # * Vertical flip: W turns into M
    #
    # Some augmentation strategies that are typically OK:
    #
    # Horizontal flip
    # Rotation
    # Contrast adjustments

    net = ConvNet(num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # training
    epochs = 1
    for epoch in range(epochs):
        total_loss = 0.0
        for features, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(dataloader_train)
        print(f"epoch {epoch + 1}, loss: {epoch_loss:.4f}")

    # evaluation
    #
    # In binary classification, we use precision / recall / f1
    #
    # Precision
    #
    # True Positives / (True Positives + False Positives)
    #
    # "Of all the positive predictions, how many were actually positive?"
    #
    # High precision: the model avoids false positives. If it predicts true,
    # it's probably true.

    # Recall
    #
    # True Positives / True Positives + False Negatives
    #
    # "Of all the instances that were actually positive, how many did the model
    # predict as positive?"
    #
    # High recall - correctly predicts true positives.
    #
    # Low recall indicates the model misses a lot of positive instances. Many
    # positives are labeled as negatives.
    #

    # With multi-class, each class has a precision and a recall.
    #
    # We can analyze them per class, or in aggregate.
    #
    # Micro average: global calculation. Calcuates precision / recall globally
    # across all classes. Computes a global precision / recall using all classes
    #
    # Macro: Computes precision / recall for each class, using a mean across all
    # classes.
    #
    # Weighted average: Computes precision / recall for each class, using a
    # *weighted* mean across all classes. Larger classes have a greater impact
    # on the final result.
    #
    # When should we use different recall types?
    #
    # * Micro: Use with imbalanced data sets as it takes into account class
    #   imbalance.
    # * Macro: Use when you care about performance of a small class (all classes
    #   treated equally)
    # * Weighted: When you consider errors in larger classes more important
    #

    test_transformers = torchvision.transforms.Compose(
        [
            # do not augument test data (of course :)
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((64, 64)),
        ]
    )

    dataset_test = torchvision.datasets.ImageFolder(
        "data/clouds/clouds_test",
        transform=test_transformers,
    )
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

    precision = torchmetrics.Precision(
        task="multiclass",
        num_classes=7,
        average="macro",
    )

    recall = torchmetrics.Recall(
        task="multiclass",
        num_classes=7,
        average="macro",
    )

    recall_per_class = torchmetrics.Recall(
        task="multiclass", num_classes=7, average="none"
    )

    net.eval()  # disable autograd for the model
    with torch.no_grad():
        for features, labels in dataloader_test:
            outputs = net(features)
            # preds are the indices (class) where the max value was found
            _, preds = torch.max(input=outputs, dim=1)
            precision(preds, labels)
            recall(preds, labels)
            recall_per_class(preds, labels)

    final_precision = precision.compute()
    final_recall = recall.compute()
    print(f"Precision: {final_precision} Recall: {final_recall}")

    final_recall_per_class = recall_per_class.compute()
    print(f"Recall per class: {final_recall_per_class}")

    # mapping of class name to index
    recall_by_class_name = {
        k: final_recall_per_class[v].item()
        for k, v in dataset_test.class_to_idx.items()
    }
    print(recall_by_class_name)
