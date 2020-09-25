import argparse
import os
import os.path
from PIL import Image, ImageOps
from torch import device, Tensor
from torchvision.datasets import VisionDataset
from typing import List, Tuple, Any
import torchvision.transforms.functional as F


# Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py


EXTENSIONS = {'jpg', 'jpeg', 'png'}
DATASET_PATH = 'datasets'
TARGETS_PATH = 'targets'
SAMPLES_PATH = 'samples'


def get_image_name(filename: str, suffix: str = None) -> str:
    tokens = filename.split('.')
    tokens[-1] = '.png'
    if suffix is not None:
        tokens[-2] += '_' + suffix

    return ''.join(tokens)


def get_centre_crop_box(width: int, height: int, target_width: int, target_height: int) -> Tuple[int, int, int, int]:
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2

    return left, top, right, bottom


def preprocess(root: os.path, targets: os.path, samples: os.path, target_width: int, target_height: int) -> None:
    try:
        os.makedirs(str(targets))
        os.makedirs(str(samples))
    except FileExistsError:
        pass

    for file in os.scandir(root):
        if not file.is_dir() and file.name.split('.')[-1] in EXTENSIONS:
            with open(file, 'rb') as f:
                img = Image.open(f)
                width, height = img.size
                if height > width:
                    img = img.transpose(Image.ROTATE_90)
                    width, height = img.size

                if width >= target_width and height >= target_height:
                    target_image = img.crop(get_centre_crop_box(width, height, target_width, target_height))
                    sample_image = ImageOps.scale(target_image, factor=0.25)

                    target_image.save(os.path.join(targets, get_image_name(file.name)), format='png')
                    sample_image.save(os.path.join(samples, get_image_name(file.name, 'sample')), format='png')

                    os.remove(file.path)


def make_dataset(targets_path: os.path, samples_path: os.path) -> List[Tuple[Any, Any]]:
    instances = []
    for root, _, fnames in sorted(os.walk(targets_path, followlinks=False)):
        for fname in sorted(fnames):
            if fname.split('.')[-1] in EXTENSIONS:
                target_path = os.path.join(root, fname)
                sample_path = os.path.join(samples_path, get_image_name(fname, 'sample'))
                instances.append((target_path, sample_path))

    return instances


def load_tensor(path: str, load_device: device) -> Tensor:
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img_tensor = F.to_tensor(img)
        return img_tensor.to(device=load_device)


class SRDataset(VisionDataset):
    TARGET, SAMPLE = 0, 1

    def __init__(self, root: str, load_device: device = device('cpu')) -> None:
        super(SRDataset, self).__init__(root)

        root_path = os.path.join(DATASET_PATH, root)

        if not os.access(root_path, os.R_OK):
            raise FileNotFoundError('Dataset folder {:s} not found in ./{:s}'.format(root, DATASET_PATH))

        targets_path = os.path.join(root_path, TARGETS_PATH)
        samples_path = os.path.join(root_path, SAMPLES_PATH)

        if not (os.access(targets_path, os.R_OK) or os.access(samples_path, os.R_OK)):
            raise FileNotFoundError('samples and targets not found in dataset folder, {:s}. Please run script data.py '
                                    'to preprocess data'.format(root))

        self.load_device = load_device

        self.image_paths = make_dataset(targets_path, samples_path)
        if len(self.image_paths) == 0:
            msg = "Found 0 files in {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(",".join(EXTENSIONS))
            raise RuntimeError(msg)

        self.loader = load_tensor

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        path = self.image_paths[index]
        sample = self.loader(path[SRDataset.SAMPLE], load_device=self.load_device)
        target = self.loader(path[SRDataset.TARGET], load_device=self.load_device)

        return sample, target

    def __len__(self) -> int:
        return len(self.image_paths)


if __name__ == '__main__':
    gen_params = {'num_features': 64, 'num_res_blocks': 8, 'num_dense_layers': 8, 'dense_bottleneck_size': 4}
    discriminator_params = {}
    loss_layers = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='MSCOCO', help='Dataset folder is ./datasets to preprocess')
    parser.add_argument('--width', type=int, default=512, help='Target width of downscaled images')
    parser.add_argument('--height', type=int, default=384, help='Target height of downscaled images')

    args = parser.parse_args()
    dataset = args.data
    width = args.width
    height = args.height

    root_path = os.path.join(DATASET_PATH, dataset)

    if not os.access(root_path, os.R_OK):
        raise FileNotFoundError('Dataset folder {:s} not found in ./{:s}'.format(dataset, DATASET_PATH))

    targets_path = os.path.join(root_path, TARGETS_PATH)
    samples_path = os.path.join(root_path, SAMPLES_PATH)

    if (width % 4) or (height % 4):
        raise ValueError('Target width or target height not divisible by upscaling factor, 4')

    if not (os.access(targets_path, os.R_OK) or os.access(samples_path, os.R_OK)):
        preprocess(root_path, targets_path, samples_path, width, height)
