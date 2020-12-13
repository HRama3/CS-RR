from modules.inception.Inception import InceptionCoding
import numpy as np
from scipy import linalg
from skimage import metrics
import torch
from torch import Tensor
from torchvision import transforms

__all__ = ['norm_inplace', 'to_numpy_uint8_img', 'peak_signal_to_noise', 'struct_similarity', 'FrechetInception']


def norm_inplace(images: Tensor) -> None:
    """
    Normalise images of batch, tensor to [0, 1]
    :param images: Tensor
    """
    assert images.ndim == 4
    with torch.no_grad():
        for i in range(images.shape[0]):
            image_view = images.select(dim=0, index=i).view(images.shape[1], -1)

            image_min, _ = torch.min(image_view, dim=1, keepdim=True)
            image_view.sub_(image_min)
            image_max, _ = torch.max(image_view, dim=1, keepdim=True)
            image_view.div_(image_max)


def to_numpy_uint8_img(images: Tensor) -> np.ndarray:
    """

    :param images:
    :return:
    """
    with torch.no_grad():
        norm_inplace(images)
        images = torch.mul(images, 255.0)

        images = images.to(dtype=torch.uint8, device='cpu')
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))

        return images_np


def peak_signal_to_noise(sr_images: Tensor, target_images: Tensor) -> Tensor:
    """

    :param sr_images:
    :param target_images:
    :return:
    """
    sr_images = to_numpy_uint8_img(sr_images)
    target_images = to_numpy_uint8_img(target_images)
    psnr = torch.empty(sr_images.shape[0], device='cpu')
    for i in range(sr_images.shape[0]):
        psnr[i] = metrics.peak_signal_noise_ratio(target_images[i, :, :, :], sr_images[i, :, :, :])

    return psnr


def struct_similarity(sr_images: Tensor, target_images: Tensor) -> Tensor:
    """

    :param sr_images:
    :param target_images:
    :return:
    """
    sr_images = to_numpy_uint8_img(sr_images)
    target_images = to_numpy_uint8_img(target_images)
    ssim = torch.empty(sr_images.shape[0], device='cpu')
    for i in range(sr_images.shape[0]):
        ssim[i] = metrics.structural_similarity(sr_images[i, :, :, :], target_images[i, :, :, :], multichannel=True)

    return ssim


class FrechetInception:
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:
        """

        :param device:
        """
        self.inception = InceptionCoding().to(device)
        self.fake_codes = torch.tensor([], device='cpu')
        self.real_codes = torch.tensor([], device='cpu')

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)

    def add_codes(self, fake_images: Tensor, real_images: Tensor) -> None:
        """

        :param fake_images: Tensor
        :param real_images: Tensor
        """

        assert fake_images.shape == real_images.shape

        with torch.no_grad():
            fakes_normed = []
            reals_normed = []
            for idx in range(fake_images.shape[0]):
                fakes_normed.append(torch.unsqueeze(self.norm(fake_images[idx, :, :, :]), dim=0))
                reals_normed.append(torch.unsqueeze(self.norm(real_images[idx, :, :, :]), dim=0))
            fake_images = torch.cat(fakes_normed, dim=0)
            real_images = torch.cat(reals_normed, dim=0)

            _fake_codes = self.inception.forward(fake_images)
            _real_codes = self.inception.forward(real_images)

            self.fake_codes = torch.cat((self.fake_codes, _fake_codes.to(device='cpu')))
            self.real_codes = torch.cat((self.real_codes, _real_codes.to(device='cpu')))

    def compute_distance(self) -> float:
        """

        :return: FID: float
        """
        fake_codes_np = self.fake_codes.numpy()
        real_codes_np = self.real_codes.numpy()

        mean_fake = np.mean(fake_codes_np, axis=1)
        cov_fake = np.cov(fake_codes_np, rowvar=False)
        mean_real = np.mean(real_codes_np, axis=1)
        cov_real = np.cov(real_codes_np, rowvar=False)

        mean_norm = np.sum(np.square(mean_fake - mean_real))
        cov_sqrt = linalg.sqrtm(np.matmul(cov_fake, cov_real))
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real

        fid = mean_norm + np.trace(cov_real + cov_fake - 2.0 * cov_sqrt)

        return fid.item()

    def clear_codes(self) -> None:
        """

        """
        self.fake_codes = torch.tensor([], device='cpu')
        self.real_codes = torch.tensor([], device='cpu')
