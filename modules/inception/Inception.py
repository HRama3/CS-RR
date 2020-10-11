from os import path
import requests
import torch
from torch import nn
from torchvision import transforms
from torchvision.models.inception import Inception3, model_urls
from torch.tensor import Tensor

# Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py


class InceptionCoding(nn.Module):
    inception_url = model_urls['inception_v3_google']

    def __init__(self) -> None:
        super(InceptionCoding, self).__init__()

        inception = Inception3(transform_input=True, aux_logits=False, init_weights=False)

        package_dir, _ = path.split(__file__)
        filename = InceptionCoding.inception_url.rsplit('/', 1)[1]
        state_dict_dir = path.join(package_dir, filename)

        self.layers = nn.Sequential(*([module for module in inception.children()][:-1]))

        try:
            state_dict = torch.load(state_dict_dir)
        except FileNotFoundError:
            print('Downloading pretrained Inception-v3 model and saving to disk')

            state_dict_pth = requests.get(InceptionCoding.inception_url)
            open(state_dict_dir, 'wb').write(state_dict_pth.content)

            state_dict = torch.load(state_dict_dir)

        self.load_state_dict(state_dict, strict=False)
        self.eval()

    def forward(self, input: Tensor) -> Tensor:
        x = self.layers(input)
        return torch.flatten(x, 1)
