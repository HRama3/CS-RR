from os import path
import requests
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.inception import Inception3, model_urls
from torch.tensor import Tensor

# Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py


class InceptionCoding(Inception3):
    inception_url = model_urls['inception_v3_google']

    def __init__(self) -> None:
        super(InceptionCoding, self).__init__(transform_input=True, aux_logits=False)

        package_dir, _ = path.split(__file__)
        filename = InceptionCoding.inception_url.rsplit('/', 1)[1]
        state_dict_dir = path.join(package_dir, filename)

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)

        try:
            state_dict = torch.load(state_dict_dir)
        except FileNotFoundError:
            print('Downloading pretrained Inception-v3 model and saving to disk')

            state_dict_pth = requests.get(InceptionCoding.inception_url)
            open(state_dict_dir, 'wb').write(state_dict_pth.content)

            state_dict = torch.load(state_dict_dir)

        self.load_state_dict(state_dict, strict=False)
        self.eval()

    def forward(self, _input: Tensor) -> Tensor:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(_input)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=False)
        # N x 2048
        x = torch.flatten(x, 1)

        return x
