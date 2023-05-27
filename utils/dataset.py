import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Subset
from PIL import Image
from torchvision import datasets

import numpy as np


def get_loaders(name="", batch_size=100, **kwargs):
    if name == "cifar10":
        return cifar10(batch_size=batch_size, **kwargs)
    elif name == "cifar100":
        return cifar100(batch_size=batch_size, **kwargs)
    elif name == "cifar10c":
        return cifar10c(batch_size=batch_size, **kwargs)
    elif name == "cifar100c":
        return cifar100c(batch_size=batch_size, **kwargs)
    elif name == "imagenet":
        return imagenet(batch_size=batch_size, **kwargs)
    elif name == "imagenetc":
        return imagenet_c(batch_size=batch_size, **kwargs)
    else:
        raise NotImplementedError


def cifar10(
    data_root="../data",
    batch_size=100,
    random_seed=508,
    num_workers=2,
    aug_level=1,
    train_no_aug=False,
):
    if train_no_aug:
        # when extracting features
        print("No aug.")
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # train. val split
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )

    valset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_test
    )

    indices = list(range(50000))
    split = 5000
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers
    )

    # test loader
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return train_loader, valid_loader, test_loader


def cifar10c(data_root="../data", batch_size=100, cname="natural", severity=1):
    assert severity in [1, 2, 3, 4, 5]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if cname == "natural":
        dataset = datasets.CIFAR10(
            os.path.join(data_root, "cifar10"),
            train=False,
            transform=transform,
            download=True,
        )
    else:
        dataset = CIFAR10C(
            os.path.join(data_root, "CIFAR-10-C"), cname, severity, transform=transform
        )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return loader


def cifar100(
    data_root="../data",
    batch_size=100,
    random_seed=508,
    num_workers=2,
    train_no_aug=False,
):
    if train_no_aug:
        # when extracting features
        print("No aug.")
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    # train. val split
    trainset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_train
    )
    valset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_test
    )

    indices = list(range(50000))
    split = 5000
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    # test loader
    testset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return train_loader, valid_loader, test_loader


def cifar100c(data_root="../data", batch_size=100, cname="natural", severity=1):
    assert severity in [1, 2, 3, 4, 5]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    if cname == "natural":
        dataset = datasets.CIFAR100(
            os.path.join(data_root, "cifar100"),
            train=False,
            transform=transform,
            download=True,
        )
    else:
        dataset = CIFAR100C(
            os.path.join(data_root, "CIFAR-100-C"), cname, severity, transform=transform
        )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return loader


######## Corruption datasets
corruptions = [
    "natural",
    "gaussian_noise",
    "shot_noise",
    "speckle_noise",
    "impulse_noise",
    "defocus_blur",
    "gaussian_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "spatter",
    "saturate",
    "frost",
]


class CIFAR10C(datasets.VisionDataset):
    def __init__(
        self, root: str, name: str, severity: int, transform=None, target_transform=None
    ):
        assert name in corruptions
        print("Corruption name: ", name)
        super(CIFAR10C, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        data_path = os.path.join(root, name + ".npy")
        target_path = os.path.join(root, "labels.npy")

        self.data = np.load(data_path)
        self.targets = np.load(target_path)

        this_idx = np.arange((severity - 1) * 10000, severity * 10000)
        self.data = self.data[this_idx]
        self.targets = self.targets[this_idx]
        print("Corruption severity: ", severity)
        print("-- data len: ", len(self.data))

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


class CIFAR100C(datasets.VisionDataset):
    def __init__(
        self, root: str, name: str, severity: int, transform=None, target_transform=None
    ):
        """
        Futa: added severity.
        """
        assert name in corruptions
        print("Corruption name: ", name)
        super(CIFAR100C, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        data_path = os.path.join(root, name + ".npy")
        target_path = os.path.join(root, "labels.npy")

        self.data = np.load(data_path)
        self.targets = np.load(target_path)

        this_idx = np.arange((severity - 1) * 10000, severity * 10000)
        self.data = self.data[this_idx]
        self.targets = self.targets[this_idx]
        print("Corruption severity: ", severity)
        print("-- data len: ", len(self.data))

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)
