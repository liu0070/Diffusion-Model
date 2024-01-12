#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import logging
import shutil
import time

import coloredlogs
import torch
import torchvision

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler,Dataset
from torchvision.transforms.functional import pad

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def plot_images(images, fig_size=(64, 64)):
    """
    Draw images
    :param images: Image
    :param fig_size: Draw image size
    :return: None
    """
    plt.figure(figsize=fig_size)
    plt.imshow(X=torch.cat([torch.cat([i for i in images.cpu()], dim=-1), ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    """
    Save images
    :param images: Image
    :param path: Save path
    :param kwargs: Other parameters
    :return: None
    """
    grid = torchvision.utils.make_grid(tensor=images, **kwargs)
    image_array = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(obj=image_array)
    im.save(fp=path)

class LimitedImageFolder(Dataset):
    def __init__(self, root, max_size, transform=None):
        self.root = root
        self.max_size = max_size
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(root)
        self.samples = self._make_dataset(root, self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        logger.info(msg=f"class to idx {class_to_idx} ....")
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        images = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames)[:self.max_size]:
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
        return images

    def loader(self, path):
        # 加载图像的方法，可以根据需要进行修改
        from PIL import Image
        return Image.open(path).convert('RGB')

def get_dataset(args, distributed=False):
    """
    Get dataset

    Automatically divide labels torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    If the dataset is as follow:
        dataset_path/class_1/image_1.jpg
        dataset_path/class_1/image_2.jpg
        ...
        dataset_path/class_2/image_1.jpg
        dataset_path/class_2/image_2.jpg
        ...

    'dataset_path' is the root directory of the dataset, 'class_1', 'class_2', etc. are different categories in
    the dataset, and each category contains several image files.

    Use the 'ImageFolder' class to conveniently load image datasets with this folder structure,
    and automatically assign corresponding labels to each image.

    You can specify the root directory where the dataset is located by passing the 'dataset_path' parameter,
    and perform operations such as image preprocessing and label conversion through other optional parameters.

    About Distributed Training:
    +------------------------+                     +-----------+
    |DistributedSampler      |                     |DataLoader |
    |                        |     2 indices       |           |
    |    Some strategy       +-------------------> |           |
    |                        |                     |           |
    |-------------+----------|                     |           |
                  ^                                |           |  4 data  +-------+
                  |                                |       -------------->+ train |
                1 | length                         |           |          +-------+
                  |                                |           |
    +-------------+----------+                     |           |
    |DataSet                 |                     |           |
    |        +---------+     |      3 Load         |           |
    |        |  Data   +-------------------------> |           |
    |        +---------+     |                     |           |
    |                        |                     |           |
    +------------------------+                     +-----------+

    :param args: Parameters
    :param distributed: Whether to distribute training
    :return: dataloader
    """
    # resize_transform = torchvision.transforms.Resize(size=(args.image_size, args.image_size))
    def pad_to_square(img):
        width, height = img.size
        desired_size = args.image_size

        # Calculate padding
        pad_width = max(0, (desired_size - width) // 2)
        pad_height = max(0, (desired_size - height) // 2)

        # Perform padding
        padded_img = pad(img, padding=(pad_width, pad_height, pad_width, pad_height), fill=0)
        return padded_img

    # transforms = torchvision.transforms.Compose([
    #     # Resize input size
    #     # torchvision.transforms.Resize(80), args.image_size + 1/4 * args.image_size
    #     torchvision.transforms.Resize(size=int(args.image_size + args.image_size / 4)),
    #     # Random adjustment cropping
    #     torchvision.transforms.RandomResizedCrop(size=args.image_size, scale=(0.8, 1.0)),
    #     # To Tensor Format
    #     torchvision.transforms.ToTensor(),
    #     # For standardization, the mean and standard deviation are both 0.5
    #     torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(pad_to_square),  # 先填充为正方形
    torchvision.transforms.Resize(size=(args.image_size, args.image_size)),  # 再缩放到指定大小
    torchvision.transforms.RandomResizedCrop(size=args.image_size, scale=(0.8, 1.0)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # 创建 LimitedImageFolder 数据集实例
    max_size_per_class = 1700  # 设置每个类别加载的最大图片数量
    dataset = LimitedImageFolder(root=args.dataset_path, max_size=max_size_per_class, transform=transforms)

    # 创建数据加载器
    # dataloader = DataLoader(limited_dataset, batch_size=32, shuffle=True)
    # Load the folder data under the current path,
    # and automatically divide the labels according to the dataset under each file name
    # dataset = torchvision.datasets.ImageFolder(root=args.dataset_path, transform=transforms)
    if distributed:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True, sampler=sampler)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)
    return dataloader


def setup_logging(save_path, run_name):
    """
    Set log saving path
    :param save_path: Saving path
    :param run_name: Saving name
    :return: List of file paths
    """
    results_root_dir = save_path
    results_dir = os.path.join(save_path, run_name)
    results_vis_dir = os.path.join(save_path, run_name, "vis")
    results_tb_dir = os.path.join(save_path, run_name, "tensorboard")
    # Root folder
    os.makedirs(name=results_root_dir, exist_ok=True)
    # Saving folder
    os.makedirs(name=results_dir, exist_ok=True)
    # Visualization folder
    os.makedirs(name=results_vis_dir, exist_ok=True)
    # Visualization folder for Tensorboard
    os.makedirs(name=results_tb_dir, exist_ok=True)
    return [results_root_dir, results_dir, results_vis_dir, results_tb_dir]


def delete_files(path):
    """
    Clear files
    :param path: Path
    :return: None
    """
    if os.path.exists(path=path):
        if os.path.isfile(path=path):
            os.remove(path=path)
        else:
            shutil.rmtree(path=path)
        logger.info(msg=f"Folder '{path}' deleted.")
    else:
        logger.warning(msg=f"Folder '{path}' does not exist.")


def save_train_logging(arg, save_path):
    with open(file=f"{save_path}/train.log", mode="a") as f:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        f.write(f"{current_time}: {arg}\n")
    f.close()


def check_and_create_dir(path):
    logger.warning(msg=f"Folder '{path}' does not exist.")
    os.makedirs(name=path, exist_ok=True)
    logger.info(msg=f"Successfully create folder '{path}'.")
