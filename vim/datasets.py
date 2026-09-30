# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np

CIFAR100_DEFAULT_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_DEFAULT_STD = (0.2675, 0.2565, 0.2761)

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

def partition_dataset(n_classes, n_tasks, randomised=False, seed=0):

    task_classes = []
    n_classes_per_task = n_classes // n_tasks

    if randomised:
        np.random.seed(seed)
        classes = np.random.permutation(n_classes)
        class_map = {classes[i]: i for i in range(n_classes)}

    else:
        classes = np.array(range(n_classes))
        class_map = {i: i for i in range (n_classes)}

    for i in range(n_tasks):

        task_classes.append(classes[i * n_classes_per_task: (i + 1) * n_classes_per_task])

    return task_classes, class_map


class SeqCIFAR10(datasets.CIFAR10):
    def __init__(self, n_tasks, randomised = False, seed=  0, *args, **kwargs):
        super(SeqCIFAR10, self).__init__(*args, **kwargs)

        self.num_classes = 10
        self.n_tasks = n_tasks
        self.n_classes_per_task = self.num_classes // self.n_tasks

        # assign classes to tasks
        self.task_classes, self.class_map = partition_dataset(self.num_classes, self.n_tasks, randomised, seed)

        self.cur_task = 0

        self.cur_task_idx = None

    def __len__(self) -> int:
        return len(self.cur_task_idx) if self.cur_task_idx is not None else super(SeqCIFAR10, self).__len__()

    def set_task(self, task):
        self.cur_task = task
        self.cur_task_idx = np.where(np.isin(self.targets, self.task_classes[self.cur_task]))[0]

    def partition_transfer(self, train_dataset):

        self.task_classes, self.class_map = train_dataset.task_classes, train_dataset.class_map

    def __getitem__(self, index):
        index = self.cur_task_idx[index]
        img, target = super(SeqCIFAR10, self).__getitem__(index)

        
        return img, self.class_map[target]

class SeqCIFAR100(datasets.CIFAR100):
    def __init__(self, n_tasks, randomised = False, seed=  0, *args, **kwargs):
        super(SeqCIFAR100, self).__init__(*args, **kwargs)

        self.num_classes = 100
        self.n_tasks = n_tasks
        self.n_classes_per_task = self.num_classes // self.n_tasks

        # assign classes to tasks
        self.task_classes, self.class_map = partition_dataset(self.num_classes, self.n_tasks, randomised, seed)

        self.cur_task = 0

        self.cur_task_idx = None
        self.partition_idx = None

    def partition_transfer(self, train_dataset):

        self.task_classes, self.class_map = train_dataset.task_classes, train_dataset.class_map

    def set_partition(self, partition_idx):
        self.partition_idx = partition_idx


    def __len__(self) -> int:
        if self.cur_task_idx is not None:
            return len(self.cur_task_idx)
        elif self.partition_idx is not None:
            return len(self.partition_idx)
        else:
            return super(SeqCIFAR100, self).__len__()
    
    def set_task(self, task):
        self.cur_task = task

        self.cur_task_idx = np.where(np.isin(self.targets, self.task_classes[self.cur_task]))[0]

        # only valid if cur_task_idx is within partition_idx
        self.cur_task_idx = self.cur_task_idx[np.where(np.isin(self.cur_task_idx, self.partition_idx))]



    def __getitem__(self, index):
        index = self.cur_task_idx[index]
        img, target = super(SeqCIFAR100, self).__getitem__(index)
        return img, self.class_map[target]

class SeqImageNetR(ImageFolder):

    def __init__(self, n_tasks, randomised = False, seed=  0, *args, **kwargs):
        super(SeqImageNetR, self).__init__(*args, **kwargs)

        self.num_classes = 200
        self.n_tasks = n_tasks
        self.n_classes_per_task = self.num_classes // self.n_tasks

        # assign classes to tasks
        self.task_classes, self.class_map = partition_dataset(self.num_classes, self.n_tasks, randomised, seed)

        self.cur_task = 0

        self.cur_task_idx = None

        self.partition_idx = None


    def partition_transfer(self, train_dataset):

        self.task_classes, self.class_map = train_dataset.task_classes, train_dataset.class_map

    def set_partition(self, partition_idx):
        self.partition_idx = partition_idx
        

    def __len__(self) -> int:
        if self.cur_task_idx is not None:
            return len(self.cur_task_idx)
        elif self.partition_idx is not None:
            return len(self.partition_idx)
        else:
            return super(SeqImageNetR, self).__len__()

    def set_task(self, task):
        self.cur_task = task

        self.cur_task_idx = np.where(np.isin(self.targets, self.task_classes[self.cur_task]))[0]

        # only valid if cur_task_idx is within partition_idx
        self.cur_task_idx = self.cur_task_idx[np.where(np.isin(self.cur_task_idx, self.partition_idx))]



    def __getitem__(self, index):
        index = self.cur_task_idx[index]
        img, target = super(SeqImageNetR, self).__getitem__(index)
        return img, self.class_map[target]


class SeqCaltech256(ImageFolder):

    def __init__(self, n_tasks, randomised = False, seed=  0, *args, **kwargs):
        super(SeqCaltech256, self).__init__(*args, **kwargs)

        self.num_classes = 250 # ignore last 6
        self.n_tasks = n_tasks
        self.n_classes_per_task = self.num_classes // self.n_tasks

        # assign classes to tasks
        self.task_classes, self.class_map = partition_dataset(self.num_classes, self.n_tasks, randomised, seed)

        self.cur_task = 0

        self.cur_task_idx = None

        self.partition_idx = np.where(np.isin(self.targets, np.arange(0, self.num_classes, 1)))[0] # get length of dataset for 250 classes


    def partition_transfer(self, train_dataset):

        self.task_classes, self.class_map = train_dataset.task_classes, train_dataset.class_map

    def set_partition(self, partition_idx):
        self.partition_idx = partition_idx
        

    def __len__(self) -> int:
        if self.cur_task_idx is not None:
            return len(self.cur_task_idx)
        elif self.partition_idx is not None:
            return len(self.partition_idx)
        else:
            return super(SeqCaltech256, self).__len__()

    def set_task(self, task):
        self.cur_task = task

        self.cur_task_idx = np.where(np.isin(self.targets, self.task_classes[self.cur_task]))[0]

        # only valid if cur_task_idx is within partition_idx
        self.cur_task_idx = self.cur_task_idx[np.where(np.isin(self.cur_task_idx, self.partition_idx))]



    def __getitem__(self, index):
        index = self.cur_task_idx[index]
        img, target = super(SeqCaltech256, self).__getitem__(index)
        return img, self.class_map[target]

class SeqCUB200(ImageFolder):

    def __init__(self, n_tasks, randomised = False, seed=  0, *args, **kwargs):

        super(SeqCUB200, self).__init__(*args, **kwargs)

        self.num_classes = 200
        self.n_tasks = n_tasks
        self.n_classes_per_task = self.num_classes // self.n_tasks

        # assign classes to tasks
        self.task_classes, self.class_map = partition_dataset(self.num_classes, self.n_tasks, randomised, seed)

        self.cur_task = 0

        self.cur_task_idx = None

        self.partition_idx = None


    def partition_transfer(self, train_dataset):

        self.task_classes, self.class_map = train_dataset.task_classes, train_dataset.class_map

    def set_partition(self, partition_idx):
        self.partition_idx = partition_idx
        

    def __len__(self) -> int:
        if self.cur_task_idx is not None:
            return len(self.cur_task_idx)
        elif self.partition_idx is not None:
            return len(self.partition_idx)
        else:
            return super(SeqCUB200, self).__len__()

    def set_task(self, task):
        self.cur_task = task

        self.cur_task_idx = np.where(np.isin(self.targets, self.task_classes[self.cur_task]))[0]

        # only valid if cur_task_idx is within partition_idx
        self.cur_task_idx = self.cur_task_idx[np.where(np.isin(self.cur_task_idx, self.partition_idx))]



    def __getitem__(self, index):
        index = self.cur_task_idx[index]
        img, target = super(SeqCUB200, self).__getitem__(index)
        return img, self.class_map[target]

class ImageNetR(ImageFolder):

    def __init__(self, *args, **kwargs):
        super(ImageNetR, self).__init__(*args, **kwargs)

        self.num_classes = 200
        self.partition_idx = None

    def __len__(self) -> int:
        if self.partition_idx is not None:
            return len(self.partition_idx)
        else:
            return super(ImageNetR, self).__len__()

    def __getitem__(self, index):
        index = self.partition_idx[index]
        img, target = super(ImageNetR, self).__getitem__(index)
        return img, target

    def set_partition(self, partition_idx):
        self.partition_idx = partition_idx


class ImageNetR_LwS(ImageNetR):

    def __init__(self, *args, **kwargs):
        super(ImageNetR_LwS, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        index = self.partition_idx[index]
        img, target = super(ImageNetR, self).__getitem__(index)
        return img, target, index



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100( args.data_path, train=is_train, transform=transform)
        nb_classes = 100

    elif args.data_set == 'SeqCIFAR100':
        dataset = SeqCIFAR100(args.n_tasks, args.random_class, args.seed, args.data_path, train=is_train, transform=transform)
        nb_classes = 100

    elif args.data_set == 'SeqCIFAR10':
        dataset = SeqCIFAR10(args.n_tasks, args.random_class, args.seed, args.data_path, train=is_train, transform=transform)
        nb_classes = 10

    elif args.data_set == 'SeqImageNetR':
        dataset = SeqImageNetR(args.n_tasks, args.random_class, args.seed, args.data_path, transform=transform)
        nb_classes = 200

    elif args.data_set == 'SeqCaltech256':
        dataset = SeqCaltech256(args.n_tasks, args.random_class, args.seed, args.data_path, transform=transform)
        nb_classes = 250

    elif args.data_set == 'SeqCUB200':
        data_path = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = SeqCUB200(args.n_tasks, args.random_class, args.seed, data_path, transform=transform)
        nb_classes = 200

    elif args.data_set == 'ImageNetR':
        dataset = ImageNetR(args.data_path, transform=transform)
        nb_classes = 200

        
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    args.n_classes_per_task = nb_classes // args.n_tasks
    args.n_classes = nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())

    if args.data_set in ['CIFAR', 'SeqCIFAR100', 'SeqCIFAR10']:
        t.append(transforms.Normalize(CIFAR100_DEFAULT_MEAN, CIFAR100_DEFAULT_STD))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)
