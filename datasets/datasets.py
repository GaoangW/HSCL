import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset, ConcatDataset
from torchvision import datasets, transforms
import lmdb
import math
from torch.utils.data import Dataset
from PIL import Image

from utils.utils import set_random_seed

DATA_PATH = '~/data/'
IMAGENET_PATH = '~/data/ImageNet'

lsun_fix_path = '/home/ps/data/LSUN_pil/LSUN_pil/'


CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class
LSUNFIX_SUPERCLASS = list(range(10))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]

class lsun_fix_dataset(Dataset): 
    def __init__(self, root, transform, target_transform=None, train=True):
        self.root = root
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        img_list = os.listdir(root)
        for n in range(len(img_list)):
            img_name = img_list[n]
            if img_name[0]!='c':
                continue
            idx1 = img_name.rfind("_")
            idx2 = img_name.rfind(".")
            img_id = int(img_name[idx1+1:idx2])
            class_id = img_id//1000

            if img_id-1000*class_id<800:
                train_flag = True
            else:
                train_flag = False

            if train_flag!=train:
                continue

            img_path = root+"/"+img_name
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = np.array(img.convert('RGB'))
                #import pdb; pdb.set_trace()
                self.data.append(img)
            
            self.targets.append(class_id)

        #import pdb; pdb.set_trace()
        self.data = np.stack(self.data)

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=True, eval=False):
    if dataset in ['imagenet', 'cub', 'stanford_dogs', 'flowers102',
                   'places365', 'food_101', 'caltech_256', 'dtd', 'pets']:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples,
                                                                                 P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet()
    else:
        train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
        #import pdb; pdb.set_trace()

    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_fix':
        '''
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        '''
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = lsun_fix_dataset(lsun_fix_path, transform=train_transform, train=True)
        test_set = lsun_fix_dataset(lsun_fix_path, transform=test_transform, train=False)
        #import pdb; pdb.set_trace()

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cifar10_imagenet':
        image_size = (32, 32, 3)
        n_classes = 10
        cifar10_train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        cifar10_test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
        ab_train = datasets.ImageFolder(DATA_PATH+"Imagenet_test/", transform=train_transform)
        ab_test = datasets.SVHN(DATA_PATH, split="test", download=download, transform=test_transform)

        
        for idx, tgt in enumerate(cifar10_train_set.targets):
            cifar10_train_set.targets[idx] = 1
        for idx, tgt in enumerate(cifar10_test_set.targets):
            cifar10_test_set.targets[idx] = 1
        

        n_classes = 10
        train_set = ConcatDataset([cifar10_train_set, ab_train])
        test_set = ConcatDataset([cifar10_test_set, ab_test])


        return train_set, test_set, image_size, n_classes, cifar10_train_set, cifar10_test_set, ab_train, ab_test

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'lsun_fix':
        return LSUNFIX_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    elif dataset == 'cifar10_imagenet':
        return [0, 1]
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        #import pdb; pdb.set_trace()
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    #import pdb; pdb.set_trace()
    return dataset

def get_subclass_with_abnormal_dataset(dataset, classes, eta=0.05):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    tgt_array = np.array(dataset.targets)
    cls_list = np.unique(tgt_array)
    N_class = len(cls_list)
    sample_num = int(np.sum(tgt_array==classes[0])*eta/(1-eta)/(N_class-1))
    cls_num = np.zeros(N_class)
    #import pdb; pdb.set_trace()
    for idx, tgt in enumerate(dataset.targets):
        #import pdb; pdb.set_trace()
        if tgt in classes:
            indices.append(idx)
        else:
            if cls_num[tgt]<sample_num:
                indices.append(idx)
                cls_num[tgt] += 1

    dataset = Subset(dataset, indices)
    #import pdb; pdb.set_trace()
    return dataset

def get_subclass_with_mixed_dataset(dataset, classes, normal_eta=0.05, polluted_eta=0.05, abnormal_eta=0.05):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    tgt_array = np.array(dataset.targets)
    cls_list = np.unique(tgt_array)
    N_class = len(cls_list)
    N_super_class = N_class//len(classes)
    total_inclass_num = 0
    for n in range(len(classes)):
        total_inclass_num += np.sum(tgt_array==classes[n])
    total_inclass_num = int(total_inclass_num)

    '''
    abnormal_num = math.ceil(total_inclass_num*abnormal_eta/(1-abnormal_eta-polluted_eta)/(N_class-len(classes)))
    polluted_num = math.ceil(total_inclass_num*polluted_eta/(1-abnormal_eta-polluted_eta)/(N_class-len(classes)))
    ab_cnt = np.zeros(N_class)
    normal_num = math.ceil(total_inclass_num*normal_eta)
    '''
    abnormal_num = round(total_inclass_num*abnormal_eta/(1-abnormal_eta)/(N_class-len(classes)))
    polluted_num = round(total_inclass_num*polluted_eta/(1-polluted_eta)/(N_class-len(classes)))
    ab_cnt = np.zeros(N_class)
    normal_num = round(total_inclass_num*normal_eta)
    
    polluted_ab_cnt = np.zeros(N_class)
    #import pdb; pdb.set_trace()
    normal_cnt = 0
    for idx, tgt in enumerate(dataset.targets):
        #import pdb; pdb.set_trace()
        if tgt in classes:
            if normal_cnt<normal_num:
                dataset.targets[idx] = dataset.targets[idx]+N_class
                normal_cnt += 1
            indices.append(idx)
        else:
            if ab_cnt[tgt]<abnormal_num:
                dataset.targets[idx] = dataset.targets[idx]+N_class
                indices.append(idx)
                ab_cnt[tgt] += 1
            elif polluted_ab_cnt[tgt]<polluted_num:
                indices.append(idx)
                polluted_ab_cnt[tgt] += 1

    dataset = Subset(dataset, indices)
    #import pdb; pdb.set_trace()
    return dataset


def get_subclass_with_mixed_unlabeled_dataset(dataset, classes, eta=0.05):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    tgt_array = np.array(dataset.targets)
    cls_list = np.unique(tgt_array)
    N_class = len(cls_list)
    abnormal_num = int(np.sum(tgt_array==classes[0])*eta/(1-eta)/(N_class-1))
    polluted_num = int(np.sum(tgt_array==classes[0])*eta/(1-eta)/(N_class-1))
    ab_cnt = np.zeros(N_class)
    normal_num = int(np.sum(tgt_array==classes[0])*eta/(1-eta))
    polluted_ab_cnt = np.zeros(N_class)
    #import pdb; pdb.set_trace()
    normal_cnt = 0
    for idx, tgt in enumerate(dataset.targets):
        #import pdb; pdb.set_trace()
        if tgt in classes:
            indices.append(idx)
        else:
            if ab_cnt[tgt]<abnormal_num:
                #dataset.targets[idx] = dataset.targets[idx]+N_class
                #indices.append(idx)
                ab_cnt[tgt] += 1
            elif polluted_ab_cnt[tgt]<polluted_num:
                indices.append(idx)
                polluted_ab_cnt[tgt] += 1

    dataset = Subset(dataset, indices)
    #import pdb; pdb.set_trace()
    return dataset

def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):

    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform


