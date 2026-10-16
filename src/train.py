"""Build a model and train it. YOU write the bodies of these functions.

The model is ResNet18, a convolutional network that was already trained on
ImageNet, 1.2 million photographs of 1,000 kinds of thing. You keep everything
it learned there and replace only its last layer, so that it answers with your
categories instead of those 1,000. Then you train it further on your images.

Run `pytest tests/test_train.py` after you fill them in. The first run
downloads the ImageNet weights, about 45 MB.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes, pretrained=True, freeze=False):
    """Return a ResNet18 whose last layer has `num_classes` outputs.

    Arguments
        num_classes  int, how many categories you have
        pretrained   True: start from the ImageNet weights,
                     torchvision.models.ResNet18_Weights.IMAGENET1K_V1.
                     False: start from random numbers, as if ImageNet had
                     never happened. You need this for the comparison in
                     Problem 3.
        freeze       True: only the new last layer is trained. Every other
                     parameter keeps the value it came with, which you do by
                     setting requires_grad to False on it.
                     False: every parameter is trained.

    Returns the model from torchvision.models.resnet18, with its `fc`
    attribute replaced by a new nn.Linear that has 512 inputs and num_classes
    outputs.

    Do not put a softmax at the end. The loss function adds it for you, and the
    web page adds it for you.
    """
    # weights=None is not the same as "any weights": it means start from random
    # numbers, which is what --scratch asks for.
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # The last layer of ResNet18 is called `fc`. It has 512 inputs, one for each
    # number the layer before it produces (the 512 features), and 1,000 outputs
    # in the ImageNet version, one per ImageNet category. Replace it, so the
    # model answers with your categories instead.
    model.fc = nn.Linear(512, num_classes)

    if freeze:
        # Freeze everything, then let the new last layer learn again. The new
        # nn.Linear was made with requires_grad True, so the second loop is what
        # makes the difference.
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return model


def train(model, X_train, y_train, X_test, y_test,
          epochs=10, lr=1e-4, batch_size=32):
    """Train the model and report what happened at every epoch.

    Arguments
        model        what build_model returned, or any other torch model
        X_train      np.float32 (N, ...)      y_train  np.int64 (N,)
        X_test       np.float32 (M, ...)      y_test   np.int64 (M,)
        epochs       int, how many times to go through the training set
        lr           float, the learning rate
        batch_size   int, how many rows per step

    Returns a dict named history with four keys. Each value is a list of
    `epochs` numbers, one per epoch:
        "train_loss"  the mean loss over the batches of that epoch
        "train_acc"   the share of training rows the model got right in those
                      batches, between 0.0 and 1.0
        "test_loss"   the mean loss over the test set, measured after the epoch
        "test_acc"    the share of the test set it gets right, measured after
                      the epoch, between 0.0 and 1.0

    What to do
        Use a GPU when there is one:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Move the model there once, and each batch there as you use it. Keep
        X_train and X_test themselves in ordinary memory: they may not fit on
        the GPU.
        Shuffle the training rows again at the start of every epoch.
        Use nn.CrossEntropyLoss, which expects raw outputs, and torch.optim.Adam
        over the parameters that have requires_grad set.
        Call model.train() before the training batches and model.eval() before
        measuring the test set. ResNet contains BatchNorm layers, which behave
        differently in the two modes.
        Do not compute gradients while measuring the test set, and never let
        the test rows change the parameters.
        Print one line per epoch, so that you can watch it while it runs.

    The model is trained in place. When this function returns, `model` is the
    trained one, and it is left on the device it was trained on.
    """
    # CPU here, GPU on Colab. The images stay in ordinary memory; only the
    # batch that is being used right now is copied to the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    # CrossEntropyLoss expects the raw numbers, and does the softmax itself. It
    # is the loss that goes with a last layer of num_classes nodes.
    criterion = nn.CrossEntropyLoss()

    # Only the parameters that may change. With freeze=True that is the new last
    # layer alone, and Adam must not be handed the frozen ones.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable, lr=lr)

    def measure(X, y):
        """Mean loss and share correct over a whole set, without gradients."""
        model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch_x = torch.from_numpy(
                    np.ascontiguousarray(X[start:start + batch_size])
                ).to(device)
                batch_y = torch.from_numpy(y[start:start + batch_size]).to(device)
                logits = model(batch_x)
                # loss is the mean over this batch: multiply it back up to get
                # the sum, and divide by the number of rows at the end.
                total_loss += float(criterion(logits, batch_y).item()) * len(batch_x)
                correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        return total_loss / len(X), correct / len(X)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        # BatchNorm behaves differently in the two modes, so say which one.
        model.train()

        # A new random order every epoch, so the batches differ and the model
        # cannot learn the order of the files.
        order = np.random.permutation(len(X_train))
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            batch_x = torch.from_numpy(
                np.ascontiguousarray(X_train[index])
            ).to(device)
            batch_y = torch.from_numpy(y_train[index]).to(device)

            optimiser.zero_grad()          # forget the previous batch
            logits = model(batch_x)        # what the model says now
            loss = criterion(logits, batch_y)
            loss.backward()                # work out the gradients
            optimiser.step()               # nudge every trainable parameter

            total_loss += float(loss.item()) * len(index)
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())

        train_loss = total_loss / len(order)
        train_acc = correct / len(order)
        test_loss, test_acc = measure(X_test, y_test)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"epoch {epoch:3d}/{epochs}  "
              f"train loss {train_loss:.4f}  train acc {train_acc:.1%}  "
              f"test loss {test_loss:.4f}  test acc {test_acc:.1%}")

    return history


def count_trainable(model):
    """How many parameters training is allowed to change. Written for you."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(model, path):
    """Save a trained model. Written for you."""
    torch.save(model, path)


def load(path):
    """Load a model saved by `save`, onto the CPU. Written for you."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model
