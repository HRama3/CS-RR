from .DenseBlock import DenseBlock
import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    def __init__(self, num_features: int, num_dense_layers: int, bottleneck_size: int, instance_norm: bool = True):
        """

        :param num_features: int
        :param num_dense_layers: int
        :param bottleneck_size: int
        :param instance_norm: bool
        """
        super(ResidualBlock, self).__init__()

        if num_features % num_dense_layers:
            raise ValueError('Non-Integral Dense Layer Growth Rate')

        self.dense_layers = DenseBlock(num_features, num_dense_layers, num_features // num_dense_layers,
                                       bottleneck_size, instance_norm)

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input: Tensor
        :return: Tensor
        """
        dense_features = self.dense_layers.forward(input)
        output = torch.add(input, dense_features)

        return output
