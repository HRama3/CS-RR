import os
import os.path
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.datasets import VisionDataset
from typing import List, Tuple, Any
import torchvision.transforms.functional as F


# Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py


EXTENSIONS = {'jpg', 'jpeg', 'png'}
DATASET_PATH = 'datasets'
TARGETS_PATH = 'targets'
SAMPLES_PATH = 'samples'


def get_sample_name(filename: str) -> str:
    tokens = filename.split('.')
    tokens[-1] = '.png'
    tokens[-2] += '_sample'

    return ''.join(tokens)


def pre_process(root: os.path, targets: os.path, samples: os.path) -> None:
    try:
        os.makedirs(str(targets))
        os.makedirs(str(samples))
    except FileExistsError:
        pass

    for file in os.scandir(root):
        if not file.is_dir():
            if file.name.split('.')[-1] in EXTENSIONS:
                with open(file, 'rb') as f:
                    img = Image.open(f)
                    img = ImageOps.scale(img, factor=0.25)
                    img.save(os.path.join(samples, get_sample_name(file.name)), format='png')
                os.rename(file.path, str(os.path.join(targets, file.name)))


def make_dataset(root: str) -> List[Tuple[Any, Any]]:
    root_path = os.path.join(DATASET_PATH, root)

    if not os.access(root_path, os.R_OK):
        raise FileNotFoundError('Dataset folder {:s} not found in ./{:s}'.format(root, DATASET_PATH))

    targets_path = os.path.join(root_path, TARGETS_PATH)
    samples_path = os.path.join(root_path, SAMPLES_PATH)

    if not (os.access(targets_path, os.R_OK) or os.access(samples_path, os.R_OK)):
        pre_process(root_path, targets_path, samples_path)

    instances = []
    for root, _, fnames in sorted(os.walk(targets_path, followlinks=False)):
        for fname in sorted(fnames):
            if fname.split('.')[-1] in EXTENSIONS:
                target_path = os.path.join(root, fname)
                sample_path = os.path.join(samples_path, get_sample_name(fname))
                instances.append((target_path, sample_path))

    return instances


def tensor_loader(path: str) -> Tensor:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.convert('RGB')
        return F.to_tensor(img)


class SRDataset(VisionDataset):
    TARGET, SAMPLE = 0, 1

    def __init__(self, root: str) -> None:
        super(SRDataset, self).__init__(root)

        image_paths = make_dataset(self.root)
        if len(image_paths) == 0:
            msg = "Found 0 files in {}\n".format(self.root)
            if EXTENSIONS is not None:
                msg += "Supported extensions are: {}".format(",".join(EXTENSIONS))
            raise RuntimeError(msg)

        self.image_paths = image_paths
        self.loader = tensor_loader

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        path = self.image_paths[index]
        sample = self.loader(path[SRDataset.SAMPLE])
        target = self.loader(path[SRDataset.TARGET])

        return sample, target

    def __len__(self) -> int:
        return len(self.image_paths)
