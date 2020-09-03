from torch import cat, nn, Tensor


class _DenseLayer(nn.Module):
    def __init__(self, growth_rate: int, num_features: int, bottleneck_size: int):
        """

        :param growth_rate: int
        :param num_features: int
        :param bottleneck_size: int
        """
        super(_DenseLayer, self).__init__()
        self.num_features = num_features

        self.bottleneck_conv = nn.Conv2d(in_channels=num_features, out_channels=bottleneck_size * growth_rate,
                                         kernel_size=1, stride=1, bias=False)
        self.bottleneck_activ = nn.ReLU(inplace=False)
        self.bottleneck_norm = nn.InstanceNorm2d(num_features=bottleneck_size * growth_rate, affine=True)

        self.dense_conv = nn.Conv2d(in_channels=bottleneck_size * growth_rate, out_channels=growth_rate,
                                    kernel_size=3, stride=1, bias=False, padding=1, padding_mode='reflect')
        self.dense_activ = nn.ReLU(inplace=False)
        self.dense_norm = nn.InstanceNorm2d(num_features=growth_rate, affine=True)

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input: Tensor
        :return: Tensor
        """

        if input.shape[1] != self.num_features:
            raise ValueError('Incorrect dimensions')

        bottleneck_features = self.bottleneck_norm(
            self.bottleneck_activ(
                self.bottleneck_conv(input)
            )
        )

        out_features = self.dense_norm(
            self.dense_activ(
                self.dense_conv(bottleneck_features)
            )
        )

        return out_features


class DenseBlock(nn.ModuleList):
    def __init__(self, num_layers: int, growth_rate: int, num_features: int, bottleneck_size: int):
        """

        :param num_layers: int
        :param growth_rate: int
        :param num_features: int
        :param bottleneck_size: int
        """
        self.dense_layers = [_DenseLayer(growth_rate, num_features, bottleneck_size)]

        for i in range(1, num_layers):
            num_features += growth_rate
            self.dense_layers.append(_DenseLayer(growth_rate, num_features, bottleneck_size))

        super(DenseBlock, self).__init__(self.dense_layers)

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input: Tensor
        :return: Tensor
        """

        output = self.dense_layers[0].forward(input)
        input = cat((input, output), dim=1)

        for layer in self.dense_layers[1:]:
            dense_features = layer.forward(input)
            output = cat((output, dense_features), dim=1)
            input = cat((input, dense_features), dim=1)

        return output