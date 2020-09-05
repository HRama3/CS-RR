from modules.blocks.ResidualBlock import ResidualBlock
from modules.layers.SubPixelConv import SubPixelConv
import torch
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    kernel_size = 3
    upscale_factor = 2

    def __init__(self, num_features: int, num_res_blocks: int, num_dense_layers: int, dense_bottleneck_size: int,
                 instance_norm: bool = True):
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
                               kernel_size=Generator.kernel_size, padding=Generator.kernel_size // 2, padding_mode='reflect')
        self.norm2 = (nn.InstanceNorm2d(num_features, affine=True) if instance_norm
                      else nn.BatchNorm2d(num_features, affine=True))
        self.activ2 = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_dense_layers, dense_bottleneck_size, instance_norm)
              for i in range(num_res_blocks)])

        self.subpixel_conv1 = SubPixelConv(num_features, num_features, Generator.upscale_factor)
        self.subpixel_conv2 = SubPixelConv(num_features // (Generator.upscale_factor ** 2), 12,
                                           Generator.upscale_factor)

        self.out_conv = nn.Conv2d(3, 3, kernel_size=Generator.kernel_size, bias=False, padding_mode='reflect')
        self.out_activ = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
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
        elwise_sum = torch.add(input2, residual)

        upscale1 = self.subpixel_conv1(elwise_sum)
        upscale2 = self.subpixel_conv2(upscale1)

        output: Tensor = self.out_activ(
            self.out_conv(upscale2)
        )

        return output
