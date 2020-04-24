from sodium.base import BaseDataLoader

from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

import os
import os.path
import hashlib
import gzip
import errno
import tarfile
import zipfile
from tqdm.auto import tqdm
import glob

from .util import download_and_extract_archive


class MNISTDataLoader:

    def __init__(self, transforms, data_dir, batch_size=64, shuffle=True, nworkers=2, pin_memory=True):
        self.data_dir = data_dir

        self.train_set = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.build_transforms(train=True)
        )

        self.test_set = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        )

        self.init_kwargs = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'num_workers': nworkers,
            'pin_memory': pin_memory
        }

    def get_loaders(self):
        return DataLoader(self.train_set, **self.init_kwargs), DataLoader(self.test_set, **self.init_kwargs)


class TinyImageNetDataLoader:

    def __init__(self, transforms, data_dir, batch_size=64, shuffle=True, nworkers=2, pin_memory=True):
        self.data_dir = data_dir

        self.train_set = TinyImageNet(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.build_transforms(train=True)
        )

        self.test_set = TinyImageNet(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        )

        self.init_kwargs = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'num_workers': nworkers,
            'pin_memory': pin_memory
        }

    def get_loaders(self):
        return DataLoader(self.train_set, **self.init_kwargs), DataLoader(self.test_set, **self.init_kwargs)


class TinyImageNet(Dataset):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    dataset_folder_name = 'tiny-imagenet-200'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.split_dir = 'train' if train else 'val'
        self.split_dir = os.path.join(
            self.root, self.dataset_folder_name, self.split_dir)
        self.image_paths = sorted(glob.iglob(os.path.join(
            self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))

        self.target = []
        self.labels = {}

        # build class label - number mapping
        with open(os.path.join(self.root, self.dataset_folder_name, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip()
                                       for text in fp.readlines()])
        self.label_text_to_number = {
            text: i for i, text in enumerate(self.label_texts)}

        # build labels for NUM_IMAGES_PER_CLASS images
        if train:
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels[f'{label_text}_{cnt}.{EXTENSION}'] = i

        # build the validation dataset
        else:
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        self.target = [self.labels[os.path.basename(
            filename)] for filename in self.image_paths]

        if download:
            self.download()

    def download(self):
        download_and_extract_archive(
            self.url, self.root, filename=self.filename)

    def __getitem__(self, index):
        filepath = self.image_paths[index]
        img = Image.open(filepath)
        img = img.convert("RGB")
        target = self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image_paths)


class CIFAR10DataLoader:

    class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, transforms, data_dir, batch_size=64, shuffle=True, nworkers=2, pin_memory=True):
        self.data_dir = data_dir

        self.train_set = datasets.CIFAR10(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.build_transforms(train=True)
        )

        self.test_set = datasets.CIFAR10(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        )

        self.init_kwargs = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'num_workers': nworkers,
            'pin_memory': pin_memory
        }

    def get_loaders(self):
        return DataLoader(self.train_set, **self.init_kwargs), DataLoader(self.test_set, **self.init_kwargs)
