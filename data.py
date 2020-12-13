import argparse
import os
import os.path
from PIL import Image, ImageFile, ImageOps
import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from typing import List, Tuple, Any

# Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py


EXTENSIONS = {'jpg', 'jpeg', 'png'}
DATASET_DIR = 'datasets'
TARGETS_DIR = 'targets'
SAMPLES_DIR = 'samples'

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_image_name(filename: str, suffix: str = None) -> str:
    tokens = filename.split('.')
    tokens[-1] = '.png'
    if suffix is not None:
        tokens[-2] += '_' + suffix

    return ''.join(tokens)


def preprocess(root: os.path, targets: os.path, samples: os.path,
               target_width: int, target_height: int, no_process: bool = False) -> None:
    try:
        os.makedirs(str(targets))
        os.makedirs(str(samples))
    except FileExistsError:
        pass

    for dirpath, _, filenames in os.walk(root):
        directory = os.path.split(dirpath)[-1]
        if not (directory == TARGETS_DIR or directory == SAMPLES_DIR):
            for filename in filenames:
                if (filename.split('.')[-1]).lower() in EXTENSIONS:
                    file = os.path.join(dirpath, filename)
                    with open(file, 'rb') as f:
                        img = Image.open(f)
                        width, height = img.size
                        if height > width and not no_process:
                            img = img.transpose(Image.ROTATE_90)
                            width, height = img.size

                        if (width >= target_width and height >= target_height) or no_process:
                            if no_process:
                                target_image = img
                            else:
                                target_image = F.center_crop(img, (target_height, target_width))
                            sample_image = ImageOps.scale(target_image, factor=0.25)

                            try:
                                sample_image.save(os.path.join(samples, get_image_name(filename, 'sample')),
                                                  format='png')
                                target_image.save(os.path.join(targets, get_image_name(filename)), format='png')
                            except OSError:
                                pass

                            os.remove(file)


def make_dataset(targets_path: os.path, samples_path: os.path) -> List[Tuple[Any, Any]]:
    instances = []
    for root, _, fnames in os.walk(targets_path, followlinks=False):
        for fname in fnames:
            if fname.split('.')[-1] in EXTENSIONS:
                target_path = os.path.join(root, fname)
                sample_path = os.path.join(samples_path, get_image_name(fname, 'sample'))
                if os.access(sample_path, os.R_OK):
                    instances.append((target_path, sample_path))

    return instances


def load_tensor(path: str, load_device: torch.device) -> Tensor:
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img_tensor = F.to_tensor(img)
        img_tensor = img_tensor.to(device=load_device)
        img_tensor.mul_(2.0)
        img_tensor.sub_(1.0)

        return img_tensor


class SRDataset(VisionDataset):
    TARGET, SAMPLE = 0, 1

    def __init__(self, root: str, load_device: torch.device = torch.device('cpu'), sort: bool = True) -> None:
        super(SRDataset, self).__init__(root)

        root_path = os.path.join(DATASET_DIR, root)

        if not os.access(root_path, os.R_OK):
            raise FileNotFoundError('Dataset folder {:s} not found in ./{:s}'.format(root, DATASET_DIR))

        targets_path = os.path.join(root_path, TARGETS_DIR)
        samples_path = os.path.join(root_path, SAMPLES_DIR)

        if not (os.access(targets_path, os.R_OK) or os.access(samples_path, os.R_OK)):
            raise FileNotFoundError('samples and targets not found in dataset folder, {:s}. Please run script data.py '
                                    'to preprocess data'.format(root))

        self.load_device = load_device

        self.image_paths = make_dataset(targets_path, samples_path)
        if len(self.image_paths) == 0:
            raise FileNotFoundError('Found 0 files in {:s}. Supported extensions are {:s}'
                                    .format(str(self.root), ','.join(EXTENSIONS)))

        if sort:
            sorted(self.image_paths, key=lambda x: os.path.split(x[1])[-1])

        self.loader = load_tensor

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        path = self.image_paths[index]
        sample = self.loader(path[SRDataset.SAMPLE], load_device=self.load_device)
        target = self.loader(path[SRDataset.TARGET], load_device=self.load_device)

        return sample, target

    def __len__(self) -> int:
        return len(self.image_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='ImageNet', help='Dataset folder in ./datasets to preprocess')
    parser.add_argument('--width', type=int, default=256, help='Target width of downscaled images')
    parser.add_argument('--height', type=int, default=256, help='Target height of downscaled images')
    parser.add_argument('--no-process', action='store_true', help='Specify to not rotate and crop target images')

    args = parser.parse_args()
    dataset = args.data
    width = args.width
    height = args.height
    no_process = args.no_process

    root_path = os.path.join(DATASET_DIR, dataset)

    if not os.access(root_path, os.R_OK):
        raise FileNotFoundError('Dataset folder {:s} not found in ./{:s}'.format(dataset, DATASET_DIR))

    targets_path = os.path.join(root_path, TARGETS_DIR)
    samples_path = os.path.join(root_path, SAMPLES_DIR)

    if (width % 4) or (height % 4):
        raise ValueError('Target width or target height not divisible by upscaling factor, 4')

    preprocess(root_path, targets_path, samples_path, width, height, no_process)
