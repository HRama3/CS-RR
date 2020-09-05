import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SubPixelConv(nn.Module):
    kernel_size = 3

    def __init__(self, num_input_features: int, num_output_features: int, upscale_factor: int):
        upscale_features = upscale_factor ** 2
        if num_output_features % upscale_features:
            raise ValueError(
                'Number of convolution output features not a multiple of r^2 = {:d}'.format(upscale_factor ** 2))

        super(SubPixelConv, self).__init__()

        self.num_input_features = num_input_features

        subkernels = torch.empty(size=(num_output_features // upscale_features, num_input_features,
                                       SubPixelConv.kernel_size, SubPixelConv.kernel_size))
        for i in range(num_output_features // upscale_features):
            nn.init.orthogonal_(subkernels.select(dim=0, index=i))

        self.conv_weights = nn.Parameter(torch.repeat_interleave(subkernels, repeats=upscale_features, dim=0),
                                         requires_grad=True)

        self.pad = nn.ReflectionPad2d(padding=SubPixelConv.kernel_size // 2)
        self.pixel_shuff = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[1] != self.num_input_features:
            raise ValueError('Incorrect number of feature maps - expected {:d} but received {:d}'.format(
                self.num_input_features, input.shape[1]
            ))

        output: Tensor = self.activ(
            self.pixel_shuff(
                F.conv2d(self.pad(input), weight=self.conv_weights)
            )
        )

        return output
