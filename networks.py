from modules.blocks.ResidualBlock import ResidualBlock
from modules.layers.SubPixelConv import SubPixelConv
import math
import torch
from torch import nn
from torch import Tensor


class Generator(nn.Module):
    KERNEL_SIZE = 3
    UPSCALE_FACTOR = 2

    def __init__(self, num_features: int, num_res_blocks: int, num_dense_layers: int, instance_norm: bool) -> None:
        """

        :rtype: None
        :param num_features:
        :param num_res_blocks:
        :param num_dense_layers:
        :param instance_norm:
        """
        super(Generator, self).__init__()

        if num_features % 2:
            raise ValueError('Number of feature maps not divisible by 2')
        if num_features % num_dense_layers:
            raise ValueError('Non-integral growth rate since number of features not a multiple of number of dense '
                             'layers,')
        if num_features < 12 * (Generator.UPSCALE_FACTOR ** 2):
            raise ValueError('Number of features too few, must be at least {:d} for sub-pixel convolution layers'
                             .format(12 * (Generator.UPSCALE_FACTOR ** 2)))

        self.in_conv = nn.Conv2d(in_channels=3, out_channels=num_features, kernel_size=Generator.KERNEL_SIZE,
                                 bias=False, padding=Generator.KERNEL_SIZE // 2, padding_mode='reflect')

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_dense_layers, instance_norm)
              for i in range(num_res_blocks)]
        )

        self.res_conv = nn.Conv2d(in_channels=num_features, out_channels=num_features,
                                  kernel_size=Generator.KERNEL_SIZE, bias=False, padding=Generator.KERNEL_SIZE // 2,
                                  padding_mode='reflect')
        self.res_norm = (nn.InstanceNorm2d(num_features, affine=True) if instance_norm
                         else nn.BatchNorm2d(num_features, affine=True))
        self.res_activ = nn.PReLU(num_parameters=num_features)

        subpixel_features = (num_features * 4) // (Generator.UPSCALE_FACTOR ** 2)
        self.subpixel_conv1 = SubPixelConv(num_features, num_features * 4, Generator.UPSCALE_FACTOR)
        self.subpixel_conv2 = SubPixelConv(subpixel_features, num_features * 4, Generator.UPSCALE_FACTOR)

        self.out_conv = nn.Conv2d(in_channels=subpixel_features, out_channels=3, kernel_size=5,
                                  bias=False, padding=2, padding_mode='reflect')
        self.out_activ = nn.Tanh()

    def forward(self, _input: Tensor) -> Tensor:
        """

        :param _input: Tensor
        :return: Tensor
        """
        _input = self.in_conv(_input)

        residual = self.res_activ(
            self.res_norm(
                self.res_conv(self.res_blocks.forward(_input))
            )
        )

        global_sum = torch.add(_input, residual)

        upscale1 = self.subpixel_conv1(global_sum)
        upscale2 = self.subpixel_conv2(upscale1)

        output: Tensor = self.out_activ(
            self.out_conv(upscale2)
        )

        return output


class Discriminator(nn.Module):
    KERNEL_SIZE = 3
    LINEAR_FEATURES = 256

    def __init__(self, input_res: int, num_features_start: int, num_features_stop: int, num_stacked_layers: int,
                 lrelu_slope: float) -> None:
        """

        :param input_res:
        :param num_features_start:
        :param num_features_stop:
        :param num_stacked_layers:
        :param lrelu_slope:
        """
        super(Discriminator, self).__init__()
        if not ((input_res & (input_res - 1) == 0) and input_res != 0):  # if max_res is not a power of two
            raise ValueError('Max res, {:d} not a exact power of two.'.format(input_res))
        if not ((num_features_start & (num_features_start - 1) == 0) and num_features_start != 0):
            # if num_maps_start is not a power of two
            raise ValueError('Number of output feature maps in first conv layer, {:d} not a power of two.'
                             .format(num_features_start))
        if not ((num_features_stop & (num_features_stop - 1) == 0) and num_features_stop != 0):
            # if num_maps_stop is not a power of two
            raise ValueError('Number of output feature maps in last conv layer, {:d} not a power of two.'
                             .format(num_features_stop))
        if num_features_start > num_features_stop:
            raise ValueError('Number of feature maps in the first layer must be fewer than or equal to the number of '
                             'feature maps in the last layer.')
        downsample_res = input_res // (num_features_stop // num_features_start)
        if downsample_res <= 1:
            raise ValueError('Input image resolution {:d}x{:d} is too small to downsample {:d} times by a factor of 2.'.
                             format(input_res, input_res, int(math.log2((num_features_stop / num_features_start)) + 1)))

        self.input_res = input_res

        layers = []
        num_input_features = 3
        _stride = 1
        while num_features_start <= num_features_stop:
            layers.append(nn.Conv2d(num_input_features, num_features_start, kernel_size=Discriminator.KERNEL_SIZE,
                                    padding=Discriminator.KERNEL_SIZE // 2, stride=_stride, padding_mode='reflect'))
            layers.append(nn.LeakyReLU(lrelu_slope, inplace=True))

            for stacked_layer in range(num_stacked_layers):
                layers.append(nn.Conv2d(num_features_start, num_features_start, kernel_size=Discriminator.KERNEL_SIZE,
                                        padding=Discriminator.KERNEL_SIZE // 2, stride=1, padding_mode='reflect'))
                layers.append(nn.LeakyReLU(lrelu_slope, inplace=True))

            num_input_features = num_features_start
            num_features_start *= 2
            _stride = 2

        self.conv_layers = nn.Sequential(*layers)

        flattened_features = num_features_stop * downsample_res * downsample_res
        self.linear = nn.Linear(in_features=flattened_features, out_features=1)
        self.out_activ = nn.Sigmoid()

    def forward(self, _input: Tensor) -> Tensor:
        """

        :param _input:
        :return:
        """
        if (_input.shape[2] < self.input_res) or (_input.shape[3] < self.input_res):
            raise ValueError('Image dimensions {:d}x{:d} must be at least {:d}x{:d}'
                             .format(_input.shape[2], _input.shape[3], self.input_res, self.input_res))

        x = self.conv_layers.forward(_input)
        x = torch.reshape(x, (x.shape[0], -1))

        output = self.out_activ(self.linear(x))

        return torch.squeeze(output)
