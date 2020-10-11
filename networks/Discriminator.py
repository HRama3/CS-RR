import math
import torch
from torch import nn, Tensor


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
