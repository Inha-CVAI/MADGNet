import os
import random

import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image

class PolypImageSegDataset(Dataset) :
    def __init__(self, dataset_dir, mode='train', transform=None, target_transform=None):
        super(PolypImageSegDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_folder = 'images'
        self.label_folder = 'masks'
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        if mode == 'train':
            self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_frame.csv'.format(mode)))
        else:
            self.frame = pd.read_csv(os.path.join('/'.join(dataset_dir.split('/')[:-1]), 'PolypSegData', '{}_{}_frame.csv'.format(dataset_dir.split('/')[-1], mode)))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # if 'CVC-ClinicDB' in self.dataset_dir:
        #     image_path = os.path.join(self.dataset_dir, '/'.join(self.frame.png_image_path[idx].split('/')[1:]))
        #     label_path = os.path.join(self.dataset_dir, '/'.join(self.frame.png_mask_path[idx].split('/')[1:]))
        # elif 'Kvasir-SEG' in self.dataset_dir:
        #     image_path = os.path.join(self.dataset_dir, 'Original', self.frame.image_path[idx])
        #     label_path = os.path.join(self.dataset_dir, 'Ground Truth', self.frame.mask_path[idx])
        # elif 'CVC-300' in self.dataset_dir or 'CVC-ColonDB' in self.dataset_dir  or 'ETIS-LaribPolypDB' in self.dataset_dir:
        #     image_path = os.path.join(self.dataset_dir, 'image', self.frame.image_path[idx])
        #     label_path = os.path.join(self.dataset_dir, 'mask', self.frame.mask_path[idx])

        if self.mode == 'train':
            image_path = os.path.join(self.dataset_dir, 'TrainDataset', self.image_folder, self.frame.image_path[idx])
            label_path = os.path.join(self.dataset_dir, 'TrainDataset', self.label_folder, self.frame.mask_path[idx])
        elif self.mode == 'test':
            image_path = os.path.join(os.path.join('/'.join(self.dataset_dir.split('/')[:-1]), 'PolypSegData', 'TestDataset', 'TestDataset', self.dataset_dir.split('/')[-1], self.image_folder, self.frame.image_path[idx]))
            label_path = os.path.join(os.path.join('/'.join(self.dataset_dir.split('/')[:-1]), 'PolypSegData', 'TestDataset', 'TestDataset', self.dataset_dir.split('/')[-1], self.label_folder, self.frame.mask_path[idx]))

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.target_transform(label)

        label[label >= 0.5] = 1; label[label < 0.5] = 0

        return image, label

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)