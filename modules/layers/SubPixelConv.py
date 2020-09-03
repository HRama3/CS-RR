import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SubPixelConv(nn.Module):
    kernel_size = 3

    def __init__(self, num_input_features: int, num_output_features: int, upscale_factor: int):
        if num_output_features % (upscale_factor ** 2):
            raise ValueError(
                'Number of convolution output features not a multiple of r^2 = {:d}'.format(upscale_factor ** 2))

        super(SubPixelConv, self).__init__()

        self.num_input_features = num_input_features

        subkernels = torch.empty(size=(1, self.num_input_features, SubPixelConv.kernel_size ** 2))
        init.orthogonal_(subkernels.select(dim=0, index=0))
        subkernels = subkernels.reshape(shape=(1, self.num_input_features, SubPixelConv.kernel_size,
                                        SubPixelConv.kernel_size))

        self.conv_weights = nn.Parameter(torch.repeat_interleave(subkernels, repeats=num_output_features, dim=0),
                                         requires_grad=True)

        self.pixel_shuff = nn.PixelShuffle(upscale_factor=upscale_factor)



    def forward(self, input: Tensor) -> Tensor:
        if input.shape[1] != self.num_input_features:
            raise ValueError('Incorrect number of feature maps - expected {:d} but received {:d}'.format(
                self.num_input_features, input.shape[1]
            ))

        input_padded = F.pad(input, pad=(SubPixelConv.kernel_size//2,)*4, mode='reflect')
        input_conv = F.conv2d(input_padded, weight=self.conv_weights)

        output: Tensor = self.pixel_shuff(input_conv)

        return output
