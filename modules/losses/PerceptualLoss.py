from os import path
import requests
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg
from torch.tensor import Tensor


class _VGG(nn.Module):
    vgg_url = vgg.model_urls['vgg16']
    cfg = 'D'

    def __init__(self):
        super(_VGG, self).__init__()
        self.features = vgg.make_layers(vgg.cfgs[_VGG.cfg], batch_norm=False)

        package_dir, _ = path.split(__file__)
        filename = _VGG.vgg_url.rsplit('/', 1)[1]
        state_dict_dir = path.join(package_dir, filename)

        try:
            state_dict = torch.load(state_dict_dir)
        except FileNotFoundError:
            print('Downloading pretrained VGG feature map parameters and saving them to disk')

            state_dict_pth = requests.get(_VGG.vgg_url)
            open(state_dict_dir, 'wb').write(state_dict_pth.content)

            state_dict = torch.load(state_dict_dir)

        self.load_state_dict(state_dict, strict=False)


class PerceptualLoss(nn.Module):
    vgg_net = _VGG()
    layers = {'relu1_1': 2, 'relu1_2': 4, 'relu2_1': 7, 'relu2_2': 9, 'relu3_1': 12,
              'relu3_2': 14, 'relu3_3': 16, 'relu4_1': 19, 'relu4_2': 21, 'relu4_3': 23}

    def __init__(self, content_weight=0.75, content_layer='relu1_2', style_layer='relu3_3'):
        """

        :param content_weight: float
        :param content_layer: string
        :param style_layer: string
        """
        super(PerceptualLoss, self).__init__()

        if content_layer not in PerceptualLoss.layers.keys():
            raise ValueError('{:s} not a valid feature layer key. Layer key must be one of {:s}'
                             .format(content_layer, str(PerceptualLoss.layers.keys())))
        if style_layer not in PerceptualLoss.layers.keys():
            raise ValueError('{:s} not a valid style layer key. Layer key must be one of {:s}'
                             .format(content_layer, str(PerceptualLoss.layers.keys())))
        if PerceptualLoss.layers[style_layer] < PerceptualLoss.layers[content_layer]:
            raise ValueError('Style layer {:s} precedes content layer {:s}'.format(style_layer, content_layer))
        if not (0.0 <= content_weight <= 1.0):
            raise ValueError('Content weight must be between 0 and 1 inclusive')

        self.content_weight = content_weight
        self.content_layers = nn.Sequential(*PerceptualLoss.vgg_net.features[0:PerceptualLoss.layers[content_layer]])
        self.style_layers = nn.Sequential(*PerceptualLoss.vgg_net.features[PerceptualLoss.layers[content_layer]:
                                                                           PerceptualLoss.layers[style_layer]])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)

        self.eval()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """

        :param input: Tensor
        :param target: Tensor
        """
        if input.shape != target.shape:
            raise ValueError('Input dimensions {:s} are different from target dimensions {:s}'
                             .format(str(input.shape), str(target.shape)))

        for i in range(input.shape[0]):
            input[i, :, :, :] -= torch.min(input.select(dim=0, index=i), dim=(1, 2), keepdim=True)

            self.norm(input.select(dim=0, index=i))
            self.norm(target.select(dim=0, index=i))

        input_content = self.content_layers.forward(input)
        target_content = self.content_layers.forward(target)

        content_loss = torch.sum(torch.square(input_content - target_content), dim=(1, 2, 3))
        content_loss.div_(input.shape[1] * input.shape[2] * input.shape[3])

        input_style = self.style_layers(input_content)
        input_style = torch.reshape(input_style, (input_style.shape[0], input_style.shape[1], -1))
        input_style_t = torch.transpose(input_style, dim0=1, dim1=2)
        target_style = self.style_layers(target_content)
        target_style = torch.reshape(target_style, (target_style.shape[0], target_style.shape[1], -1))
        target_style_t = torch.transpose(target_style, dim0=1, dim1=2)

        gram_input = torch.matmul(input_style, input_style_t)
        gram_input.div_(input.shape[1] * input.shape[2] * input.shape[3])
        gram_target = torch.matmul(target_style, target_style_t)
        gram_target.div_(target.shape[1] * target.shape[2] * target.shape[3])

        style_loss = torch.norm(torch.sub(gram_input, gram_target), p='fro', dim=(1, 2))

        loss = torch.sum(torch.mul(content_loss, self.content_weight) + torch.mul(style_loss, 1.0 - self.content_weight))

        return loss
