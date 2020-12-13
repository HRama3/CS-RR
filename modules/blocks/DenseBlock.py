import torch
import torch.nn as nn
from torch import Tensor


class _DenseLayer(nn.Module):
    kernel_size = 3

    def __init__(self, num_features: int, growth_rate: int, instance_norm: bool) -> None:
        """

        :param growth_rate: int
        :param num_features: int
        :param instance_norm: bool
        """
        super(_DenseLayer, self).__init__()

        self.num_features = num_features

        self.conv = nn.Conv2d(in_channels=num_features, out_channels=growth_rate, bias=False,
                              kernel_size=_DenseLayer.kernel_size, padding=_DenseLayer.kernel_size // 2,
                              padding_mode='reflect')
        self.norm = (nn.InstanceNorm2d(num_features=growth_rate, affine=True) if instance_norm
                     else nn.BatchNorm2d(num_features=growth_rate, affine=True))
        self.activ = nn.PReLU(num_parameters=growth_rate)

    def forward(self, _input: Tensor) -> Tensor:
        """

        :param _input: Tensor
        :return: Tensor
        """

        if _input.shape[1] != self.num_features:
            raise ValueError('Incorrect channel dimension')

        output: Tensor = self.activ(
            self.norm(
                self.conv(_input)
            )
        )

        return output


class DenseBlock(nn.ModuleList):
    def __init__(self, num_features: int, num_layers: int, growth_rate: int, instance_norm: bool) -> None:
        """

        :param num_layers: int
        :param growth_rate: int
        :param num_features: int
        :param instance_norm: bool
        """
        self.dense_layers = [_DenseLayer(num_features, growth_rate, instance_norm)]

        for i in range(1, num_layers):
            num_features += growth_rate
            self.dense_layers.append(_DenseLayer(num_features, growth_rate, instance_norm))

        super(DenseBlock, self).__init__(self.dense_layers)

    def forward(self, _input: Tensor) -> Tensor:
        """

        :param _input: Tensor
        :return: Tensor
        """
        output_features = []

        for layer in self.dense_layers:
            dense_features = layer.forward(_input)
            output_features.append(dense_features)
            _input = torch.cat((_input, dense_features), dim=1)

        return torch.cat(output_features, dim=1)
