from .DenseBlock import DenseBlock
import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    def __init__(self, num_features: int, num_dense_layers: int, instance_norm: bool) -> None:
        """

        :param num_features: int
        :param num_dense_layers: int
        :param instance_norm: bool
        """
        super(ResidualBlock, self).__init__()

        if num_features % num_dense_layers:
            raise ValueError('Non-Integral Dense Layer Growth Rate')

        self.dense_layers = DenseBlock(num_features, num_dense_layers, num_features // num_dense_layers, instance_norm)

    def forward(self, _input: Tensor) -> Tensor:
        """

        :param _input: Tensor
        :return: Tensor
        """
        dense_features = self.dense_layers.forward(_input)

        output = torch.add(_input, dense_features)

        return output
