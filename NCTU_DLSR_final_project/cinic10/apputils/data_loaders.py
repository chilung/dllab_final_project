#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Helper code for data loading.

This code will help with the image classification datasets: ImageNet and CIFAR10

"""
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# ctsun 181125
DATASETS_NAMES = ['imagenet', 'cifar10', 'cinic10_npz']


def load_data(dataset, data_dir, batch_size, workers, valid_size=0.1, deterministic=False):
    """Load a dataset.

    Args:
        dataset: a string with the name of the dataset to load (cifar10/imagenet)
        data_dir: the directory where the datset resides
        batch_size: the batch size
        workers: the number of worker threads to use for loading the data
        valid_size: portion of training dataset to set aside for validation
        deterministic: set to True if you want the data loading process to be deterministic.
          Note that deterministic data loading suffers from poor performance.
    """
    assert dataset in DATASETS_NAMES
    if dataset == 'cifar10':
        return cifar10_load_data(data_dir, batch_size, workers, valid_size=valid_size, deterministic=deterministic)
    # ctsun 181125
    if dataset == 'cinic10_npz':
        return cinic10_load_data_npz(data_dir, batch_size, workers, valid_size=valid_size, deterministic=deterministic)
    if dataset == 'imagenet':
        return imagenet_load_data(data_dir, batch_size, workers, valid_size=valid_size, deterministic=deterministic)
    print("FATAL ERROR: load_data does not support dataset %s" % dataset)
    exit(1)


def __image_size(dataset):
    # un-squeeze is used here to add the batch dimension (value=1), which is missing
    return dataset[0][0].unsqueeze(0).size()


def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def cifar10_load_data(data_dir, batch_size, num_workers, valid_size=0.1, deterministic=False):
    """Load the CIFAR10 dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-1, 1]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    worker_init_fn = __deterministic_worker_init_fn if deterministic else None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    valid_loader = None
    if split > 0:
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

    testset = datasets.CIFAR10(root=data_dir, train=False,
                               download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    input_shape = __image_size(train_dataset)

    # If validation split was 0 we use the test set as the validation set
    return train_loader, valid_loader or test_loader, test_loader, input_shape


def imagenet_load_data(data_dir, batch_size, num_workers, valid_size=0.1, deterministic=False):
    """Load the ImageNet dataset.

    Somewhat unconventionally, we use the ImageNet validation dataset as our test dataset,
    and split the training dataset for training and validation (90/10 by default).
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # Note! We must shuffle the imagenet data because the files are ordered
    # by class.  If we don't shuffle, the train and validation datasets will
    # by mutually-exclusive
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    input_shape = __image_size(train_dataset)

    worker_init_fn = __deterministic_worker_init_fn if deterministic else None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    valid_loader = None
    if split > 0:
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    # If validation split was 0 we use the test set as the validation set
    return train_loader, valid_loader or test_loader, test_loader, input_shape


# ctsun 181125
from PIL import Image
import os.path
import torch.utils.data as data
import glob

class CINIC10_NPZ(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory ``cinic10_train.npz`` & 
            ``cinic10_test.npz`` exists or will be generated if autogen is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
            
    Generate NPZ file Usage:
        cinic10_npz = CINIC10_NPZ()
        cinic10_npz.save_npz()
    """
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    sets = ['train', 'valid', 'test']
    train_npz = 'cinic10_train.npz'
    test_npz = 'cinic10_test.npz'

    def __init__(self, root='./data/', train=True,
                 transform=None, target_transform=None, autogen=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.autogen = autogen
        
        if self.train:
            self.downloaded_list = self.train_npz
        else:
            self.downloaded_list = self.test_npz
            
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        file_name = self.root+self.downloaded_list
        #print('file_name:',file_name)
        if os.path.exists(file_name):
            entry = np.load(file_name)
            self.data  = entry['data']
            self.targets = entry['label']
        else:
            print('Preparing the npz dataset by manual...')
            #self.save_npz()
            print('$ te = CINIC10_NPZ()')
            print('$ te.save_npz()')
            print('Done')
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #print('index:', index)
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def save_npz(self):
        data_path = self.root+'cinic10'
        data_train, label_train = [], []
        data_test, label_test = [], []
        
        for s in self.sets:
            for c in self.classes:
                source_directory = '{}/{}/{}'.format(data_path, s, c)
                filenames = glob.glob('{}/*.png'.format(source_directory))
                i = 0
                for fn in filenames:
                    img = Image.open(fn).convert('RGB')
                    img = np.array(img)
                    if s == 'train' or s == 'valid':
                        data_train.append(img)
                        label_train.append(self.classes.index(c))

                    elif s == 'test':
                        data_test.append(img)
                        label_test.append(self.classes.index(c))
                        
        print('save training & validation dataset to ', self.root+self.train_npz)
        print('save testing dataset to ', self.root+self.test_npz)
        np.savez(self.root+self.train_npz, data=data_train, label=label_train)
        np.savez(self.root+self.test_npz, data=data_test, label=label_test)


def cinic10_load_data_npz(data_dir, batch_size, num_workers, valid_size=0.1, deterministic=False):
    """Load the CINIC10 dataset in NPZ format. """
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CINIC10_NPZ(root=data_dir, train=True, transform=transform)
    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    worker_init_fn = __deterministic_worker_init_fn if deterministic else None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    
    valid_loader = None
    if split > 0:
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

    testset = CINIC10_NPZ(root=data_dir, train=False, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    input_shape = __image_size(train_dataset)
    
    # If validation split was 0 we use the test set as the validation set
    return train_loader, valid_loader or test_loader, test_loader, input_shape
