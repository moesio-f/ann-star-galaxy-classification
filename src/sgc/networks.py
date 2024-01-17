"""Neural Network builders with
torch.
"""
from __future__ import annotations

import math
from typing import Callable, Literal

import torch
import torchvision.models as models
from torch import nn

_N_CLASSES: int = 2


def simple_cnn(input_ch: int,
               img_shape: tuple[int, int],
               kernels: list[int],
               channels: list[int],
               poolings: list[Callable[[], nn.Module]],
               paddings: list[int],
               strides: list[int] = None,
               activation_fn: Callable[[], nn.Module] = None,
               channel_first: bool = True):
    assert len(kernels) > 0
    assert len(kernels) == len(channels)
    assert len(kernels) == len(poolings)
    assert len(kernels) == len(paddings)

    # Set defaults
    if strides is None:
        strides = [1] * len(kernels)

    if activation_fn is None:
        activation_fn = nn.ReLU

    # Construct layers
    layers = []
    for i in range(len(kernels)):
        n_prev = input_ch if i <= 0 else channels[i - 1]
        padding = 0 if paddings[i] is None else paddings[i]
        hidden_layer = [nn.Conv2d(in_channels=n_prev,
                                  out_channels=channels[i],
                                  kernel_size=kernels[i],
                                  stride=strides[i],
                                  padding=padding,
                                  bias=True),
                        activation_fn()]

        # Maybe add pooling
        pooling = poolings[i]
        if pooling is not None:
            hidden_layer.append(pooling())

        # Extend layers with hidden layer
        layers.extend(hidden_layer)

    # Output layer
    layers.extend((nn.Flatten(),
                   nn.LazyLinear(out_features=_N_CLASSES,
                                 bias=True)))

    # Create MLP
    mlp = nn.Sequential(*layers)

    # Initialize lazy layers
    input_shape = (1, input_ch, *img_shape)
    if not channel_first:
        input_shape = (1, *img_shape, input_ch)
    mlp(torch.zeros(input_shape))
    return mlp


def resnet_18(head_only: bool):
    net = models.resnet18(weights='IMAGENET1K_V1')

    if head_only:
        for param in net.parameters():
            param.requires_grad = False

    net.fc = nn.Linear(
        in_features=net.fc.in_features,
        out_features=_N_CLASSES)

    return net


def densenet_121(head_only: bool):
    net = models.densenet121(weights='IMAGENET1K_V1')

    if head_only:
        for param in net.parameters():
            param.requires_grad = False

    net.classifier = nn.Linear(
        in_features=net.classifier.in_features,
        out_features=_N_CLASSES)

    return net


def vgg_16(head_only: bool):
    net = models.vgg16(weights='IMAGENET1K_V1')

    if head_only:
        for param in net.parameters():
            param.requires_grad = False

    net.classifier[-1] = nn.Linear(
        in_features=net.classifier[-1].in_features,
        out_features=_N_CLASSES)

    return net


def convnext_base(head_only: bool):
    net = models.convnext_base(weights='IMAGENET1K_V1')

    if head_only:
        for param in net.parameters():
            param.requires_grad = False

    net.classifier[-1] = nn.Linear(
        in_features=net.classifier[-1].in_features,
        out_features=_N_CLASSES)

    return net


def simple_mlp(img_shape: tuple[int, int, int],
               n_hidden: list[int],
               hidden_fn: Callable[[], nn.Module] = None):
    # Set defaults
    if hidden_fn is None:
        hidden_fn = nn.ReLU

    # Assertions
    assert len(n_hidden) > 0

    # Input layer
    layers = [nn.Flatten()]
    n_input = math.prod(img_shape)

    # Hidden layers
    for i in range(len(n_hidden)):
        n_prev = n_input if i <= 0 else n_hidden[i - 1]
        hidden_layer = [nn.Linear(n_prev,
                                  n_hidden[i],
                                  bias=True),
                        hidden_fn()]

        # Extend layers with hidden layer
        layers.extend(hidden_layer)

    # Output layer
    layers.append(nn.Linear(n_hidden[-1],
                            _N_CLASSES,
                            bias=True))

    return nn.Sequential(*layers)


def simple_rnn(img_shape: tuple[int, int, int],
               rnn_kind: Literal['rnn'] | Literal['lstm'] | Literal['gru'],
               hidden_size: int,
               num_layers: int):
    # Input size is number of columns times the
    #   number of channels
    input_size = img_shape[-1]

    # Layer that converts the img_shape
    #   to a sequence of features
    class _ReshapeImage(nn.Module):
        def forward(self, x):
            # from (batch, ch, rows, cols)
            # to   (batch, seq = ch * rows, cols)
            return x.flatten(1, 2)

    # Extra layer to remove hidden state
    class _TensorExtractor(nn.Module):
        def forward(self, x):
            tensor, _ = x
            return tensor[:, -1, :]

    # Select RNN implementation
    rnn = nn.RNN
    if rnn_kind == 'lstm':
        rnn = nn.LSTM
    elif rnn_kind == 'gru':
        rnn = nn.GRU

    # Create and return network
    return nn.Sequential(_ReshapeImage(),
                         rnn(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True),
                         _TensorExtractor(),
                         nn.Linear(hidden_size,
                                   _N_CLASSES))
