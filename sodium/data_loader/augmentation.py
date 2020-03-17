import abc

import torchvision.transforms as T
import numpy as np

import albumentations as A
import albumentations.pytorch.transforms as AT


class AlbumentationTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img = np.array(img)

        return self.transforms(image=img)['image']


class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class MNISTTransforms(AugmentationFactoryBase):

    def build_train(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))])

    def build_test(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))])


class CIFAR10Transforms(AugmentationFactoryBase):

    def build_train(self):
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def build_test(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


class CIFAR10Albumentations(AugmentationFactoryBase):

    def build_train(self):
        train_transforms = A.Compose([
            A.Rotate((-30.0, 30.0)),
            A.HorizontalFlip(),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010)),
            A.Cutout(num_holes=4)
            AT.ToTensor()
        ])

        return AlbumentationTransforms(train_transforms)

    def build_test(self):
        test_transforms = A.Compose([
            A.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010)),
            AT.ToTensor()
        ])

        return AlbumentationTransforms(test_transforms)
