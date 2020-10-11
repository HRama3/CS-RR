from modules.blocks.ResidualBlock import ResidualBlock
from modules.layers.SubPixelConv import SubPixelConv
import math
import torch
from torch import nn
from torch import Tensor


class Generator(nn.Module):
    kernel_size = 3
    upscale_factor = 2

    def __init__(self, num_features: int, num_res_blocks: int, num_dense_layers: int, dense_bottleneck_size: int,
                 instance_norm: bool = True) -> None:
        """

        :rtype: None
        :param num_features:
        :param num_res_blocks:
        :param num_dense_layers:
        :param dense_bottleneck_size:
        :param instance_norm:
        """
        super(Generator, self).__init__()

        if num_features % 2:
            raise ValueError('Number of feature maps not divisible by 2')
        if num_features % num_dense_layers:
            raise ValueError('Non-integral growth rate since number of features not a multiple of number of dense '
                             'layers,')
        if num_features < 12 * (Generator.upscale_factor ** 2):
            raise ValueError('Number of features too few, must be at least {:d} for sub-pixel convolution layers'
                             .format(12 * (Generator.upscale_factor ** 2)))

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_features // 2, kernel_size=Generator.kernel_size,
                               bias=False, padding=Generator.kernel_size // 2, padding_mode='reflect')
        self.norm1 = (nn.InstanceNorm2d(num_features // 2, affine=True) if instance_norm
                      else nn.BatchNorm2d(num_features // 2, affine=True))
        self.activ1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_features // 2, out_channels=num_features, bias=False,
                               kernel_size=Generator.kernel_size,
                               padding=Generator.kernel_size // 2, padding_mode='reflect')
        self.norm2 = (nn.InstanceNorm2d(num_features, affine=True) if instance_norm
                      else nn.BatchNorm2d(num_features, affine=True))
        self.activ2 = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_dense_layers, dense_bottleneck_size, instance_norm)
              for i in range(num_res_blocks)])

        self.subpixel_conv1 = SubPixelConv(num_features, num_features, Generator.upscale_factor)
        self.subpixel_conv2 = SubPixelConv(num_features // (Generator.upscale_factor ** 2), 12,
                                           Generator.upscale_factor)

        self.out_conv = nn.Conv2d(3, 3, kernel_size=Generator.kernel_size, bias=False, padding_mode='reflect', padding=1)
        self.out_activ = nn.ReLU(inplace=False)

    def forward(self, input: Tensor) -> Tensor:
        """

        :param input: Tensor
        :return: Tensor
        """
        input1 = self.activ1(
            self.norm1(
                self.conv1(input)
            )
        )

        input2 = self.activ2(
            self.norm2(
                self.conv2(input1)
            )
        )

        residual = self.res_blocks(input2)
        global_sum = torch.add(input2, residual)

        upscale1 = self.subpixel_conv1(global_sum)
        upscale2 = self.subpixel_conv2(upscale1)

        output: Tensor = self.out_activ(
            self.out_conv(upscale2)
        )

        return output


class Discriminator(nn.Module):
    def __init__(self, input_res: int, num_features_start: int, num_features_stop: int, num_stacked_layers: int,
                 lrelu_slope: float) -> None:
        super(Discriminator, self).__init__()
        if not((input_res & (input_res - 1) == 0) and input_res != 0):  # if max_res is not a power of two
            raise ValueError('Max res, {:d} not a exact power of two.'.format(input_res))
        if not((num_features_start & (num_features_start - 1) == 0) and num_features_start != 0):
            # if num_maps_start is not a power of two
            raise ValueError('Number of output feature maps in first conv layer, {:d} not a power of two.'
                             .format(num_features_start))
        if not((num_features_stop & (num_features_stop - 1) == 0) and num_features_stop != 0):
            # if num_maps_stop is not a power of two
            raise ValueError('Number of output feature maps in last conv layer, {:d} not a power of two.'
                             .format(num_features_stop))
        if num_features_start > num_features_stop:
            raise ValueError('Number of feature maps in the first layer must be fewer than or equal to the number of '
                             'feature maps in the last layer.')
        downsample_res = input_res // (num_features_stop // num_features_start) // 2
        if downsample_res <= 1:
            raise ValueError('Input image resolution {:d}x{:d} is too small to downsample {:d} times by a factor of 2.'.
                             format(input_res, input_res, int(math.log2((num_features_stop / num_features_start)) + 1)))

        self.input_res = input_res

        layers = []
        num_input_features = 3
        while num_features_start <= num_features_stop:
            layers.append(nn.Conv2d(num_input_features, num_features_start, kernel_size=3, stride=2, padding=1,
                                    padding_mode='reflect'))
            layers.append(nn.BatchNorm2d(num_features_start))
            layers.append(nn.LeakyReLU(lrelu_slope, inplace=True))

            for stacked_layer in range(num_stacked_layers):
                layers.append(nn.Conv2d(num_features_start, num_features_start, kernel_size=3, stride=1, padding=1,
                                        padding_mode='reflect'))
                layers.append(nn.BatchNorm2d(num_features_start))
                layers.append(nn.LeakyReLU(lrelu_slope, inplace=True))

            num_input_features = num_features_start
            num_features_start *= 2

        flattened_features = num_features_stop * downsample_res * downsample_res

        self.conv_layers = nn.Sequential(*layers)
        self.linear1 = nn.Linear(in_features=flattened_features, out_features=flattened_features // 2)
        self.linear_lrelu = nn.LeakyReLU(lrelu_slope)
        self.linear2 = nn.Linear(in_features=flattened_features // 2, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        if (input.shape[2] < self.input_res) or (input.shape[3] < self.input_res):
            raise ValueError('Image dimensions {:d}x{:d} must be at least {:d}x{:d}'
                             .format(input.shape[2], input.shape[3], self.input_res, self.input_res))

        x = self.conv_layers(input)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.linear_lrelu(self.linear1(x))

        output = self.sigmoid(self.linear2(x))
        return output
