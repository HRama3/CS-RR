import torch
import torch.nn as nn
from torch import Tensor


class _DenseLayer(nn.Module):
    kernel_size = 3

    def __init__(self, num_features: int, growth_rate: int, bottleneck_size: int, instance_norm: bool = True):
        """

        :param growth_rate: int
        :param num_features: int
        :param bottleneck_size: int
        :param instance_norm: bool
        """
        super(_DenseLayer, self).__init__()

        self.num_features = num_features

        self.bottleneck_conv = nn.Conv2d(in_channels=num_features, out_channels=bottleneck_size * growth_rate,
                                         kernel_size=1, bias=False)
        self.bottleneck_norm = (
            nn.InstanceNorm2d(num_features=bottleneck_size * growth_rate, affine=True) if instance_norm
            else nn.BatchNorm2d(num_features=bottleneck_size * growth_rate, affine=True))
        self.bottleneck_activ = nn.ReLU(inplace=True)

        self.dense_conv = nn.Conv2d(in_channels=bottleneck_size * growth_rate, out_channels=growth_rate, bias=False,
                                    kernel_size=_DenseLayer.kernel_size, padding=_DenseLayer.kernel_size//2,
                                    padding_mode='reflect')
        self.dense_norm = (nn.InstanceNorm2d(num_features=growth_rate, affine=True) if instance_norm
                           else nn.BatchNorm2d(num_features=growth_rate, affine=True))
        self.dense_activ = nn.ReLU(inplace=True)

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input: Tensor
        :return: Tensor
        """

        if input.shape[1] != self.num_features:
            raise ValueError('Incorrect channel dimensions')

        bottleneck_features = self.bottleneck_activ(
            self.bottleneck_norm(
                self.bottleneck_conv(input)
            )
        )

        output: Tensor = self.dense_activ(
            self.dense_norm(
                self.dense_conv(bottleneck_features)
            )
        )

        return output


class DenseBlock(nn.ModuleList):
    def __init__(self, num_features: int, num_layers: int, growth_rate: int,
                 bottleneck_size: int, instance_norm: bool = True):
        """

        :param num_layers: int
        :param growth_rate: int
        :param num_features: int
        :param bottleneck_size: int
        :param instance_norm: bool
        """
        self.dense_layers = [_DenseLayer(num_features, growth_rate, bottleneck_size, instance_norm)]

        for i in range(1, num_layers):
            num_features += growth_rate
            self.dense_layers.append(_DenseLayer(num_features, growth_rate, bottleneck_size, instance_norm))

        super(DenseBlock, self).__init__(self.dense_layers)

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input: Tensor
        :return: Tensor
        """
        output: Tensor = self.dense_layers[0].forward(input)
        input = torch.cat((input, output), dim=1)

        for layer in self.dense_layers[1:]:
            dense_features = layer.forward(input)
            output = torch.cat((output, dense_features), dim=1)
            input = torch.cat((input, dense_features), dim=1)

        return output
