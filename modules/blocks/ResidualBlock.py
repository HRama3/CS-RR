from .DenseBlock import DenseBlock
from torch import nn, Tensor


class ResidualBLock(nn.Module):
    def __init__(self, num_features: int, num_dense_layers: int, bottleneck_size: int):
        """

        :param num_features: int
        :param num_dense_layers: int
        :param bottleneck_size: int
        """
        super(ResidualBLock, self).__init__()

        if num_features % num_dense_layers:
            raise ValueError('Non-Integral Dense Layer Growth Rate')

        self.dense_layers = DenseBlock(num_dense_layers, num_features//num_dense_layers, num_features, bottleneck_size)

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input: Tensor
        :return: Tensor
        """
        dense_features = self.dense_layers.forward(input)
        output = input + dense_features

        return output