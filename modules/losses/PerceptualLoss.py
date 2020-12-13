import os
import requests
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg
from torch.tensor import Tensor
from typing import Tuple

# Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
VGG_DIR = 'vgg/'


class _VGG(nn.Module):
    vgg_url = vgg.model_urls['vgg19']
    cfg = 'E'

    def __init__(self) -> None:
        super(_VGG, self).__init__()
        self.features = vgg.make_layers(vgg.cfgs[_VGG.cfg], batch_norm=False)

        package_dir, _ = os.path.split(__file__)
        filename = _VGG.vgg_url.rsplit('/', 1)[1]
        state_dict_dir = os.path.join(package_dir, VGG_DIR, filename)

        try:
            state_dict = torch.load(state_dict_dir)
        except FileNotFoundError:
            print('Downloading pretrained VGG model and saving to disk')

            state_dict_path = requests.get(_VGG.vgg_url)

            if not os.path.exists(os.path.join(package_dir, VGG_DIR)):
                os.mkdir(os.path.join(package_dir, VGG_DIR))

            open(state_dict_dir, 'wb').write(state_dict_path.content)

            state_dict = torch.load(state_dict_dir)

        self.load_state_dict(state_dict, strict=False)


class PerceptualLoss(nn.Module):
    vgg_net = _VGG()
    layers = {'relu1_1': 1, 'relu1_2': 3, 'relu2_1': 6, 'relu2_2': 8, 'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
              'relu3_4': 17, 'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26, 'relu5_1': 29, 'relu5_2': 31,
              'relu5_3': 33, 'relu5_4': 35}

    def __init__(self, content_weight: float, content_layer: str, style_layer: str) -> None:
        """

        :param content_weight: float
        :param content_layer: str
        :param style_layer: str
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
                                                                           PerceptualLoss.layers[style_layer] + 1])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)

        self.eval()

    def forward(self, sr_images: Tensor, target_images: Tensor) -> Tuple[Tensor, Tensor]:
        """

        :param sr_images: Tensor
        :param target_images: Tensor
        :return: Tensor
        """
        if sr_images.shape != target_images.shape:
            raise ValueError('Input dimensions {:s} are different from target dimensions {:s}'
                             .format(str(sr_images.shape), str(target_images.shape)))
        if sr_images.shape[1] != 3:
            raise ValueError('Images in batch have {:d} channel dimensions when 3 is expected'
                             .format(sr_images.shape[1]))

        sr_normed = []
        target_normed = []
        for idx in range(sr_images.shape[0]):
            sr_normed.append(torch.unsqueeze(self.norm(sr_images[idx, :, :, :]), dim=0))
            target_normed.append(torch.unsqueeze(self.norm(target_images[idx, :, :, :]), dim=0))
        sr_images = torch.cat(sr_normed, dim=0)
        target_images = torch.cat(target_normed, dim=0)

        sr_content = self.content_layers.forward(sr_images)
        target_content = self.content_layers.forward(target_images)

        content_loss = torch.sum(torch.pow(sr_content - target_content, 2.0), dim=(1, 2, 3))
        content_loss = torch.mean(content_loss)
        content_loss = torch.mul(content_loss,
                                 self.content_weight / (sr_content.shape[1] * sr_content.shape[2] * sr_content.shape[3]))

        sr_style = self.style_layers.forward(sr_content)
        sr_style = torch.reshape(sr_style, (sr_style.shape[0], sr_style.shape[1], -1))
        sr_style_t = torch.transpose(sr_style, dim0=1, dim1=2)
        target_style = self.style_layers.forward(target_content)
        target_style = torch.reshape(target_style, (target_style.shape[0], target_style.shape[1], -1))
        target_style_t = torch.transpose(target_style, dim0=1, dim1=2)

        sr_gram = torch.matmul(sr_style, sr_style_t)
        sr_gram = torch.div(sr_gram, sr_images.shape[1] * sr_images.shape[2] * sr_images.shape[3])
        target_gram = torch.matmul(target_style, target_style_t)
        target_gram = torch.div(target_gram, target_images.shape[1] * target_images.shape[2] * target_images.shape[3])

        gram_F = torch.abs(torch.sub(sr_gram, target_gram))
        gram_F = torch.sum(torch.pow(gram_F, 2.0), dim=(1, 2))
        style_loss = torch.mul(gram_F, 1.0 - self.content_weight)
        style_loss = torch.mean(style_loss)

        return content_loss, style_loss
