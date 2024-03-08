import os
import random

import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class DataScienceBowl2018Dataset(Dataset) :
    def __init__(self, dataset_dir, mode, transform=None, target_transform=None):
        super(DataScienceBowl2018Dataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_folder = 'stage1_train'
        self.transform = transform
        self.target_transform = target_transform
        self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_imageid_frame.csv'.format(mode)))
        # data_imageId = self.frame.ImageId.drop_duplicates()
        #
        # train, test = train_test_split(data_imageId, test_size=0.10, shuffle=False, random_state=4321)
        # train, val = train_test_split(train, test_size=0.11, shuffle=False, random_state=4321)
        #
        # train = train.reset_index().drop(['index'], axis=1)
        # val = val.reset_index().drop(['index'], axis=1)
        # test = test.reset_index().drop(['index'], axis=1)
        #
        # train.to_csv(os.path.join(dataset_dir, 'train_imageid_frame.csv'))
        # val.to_csv(os.path.join(dataset_dir, 'val_imageid_frame.csv'))
        # test.to_csv(os.path.join(dataset_dir, 'test_imageid_frame.csv'))
        #
        # sys.exit()

    def __len__(self):
        return len(self.frame.ImageId.unique())

    def _get_image(self, idx):# iloc (int): The position in the dataframe
        # Collect image name from csv frame
        img_name = self.frame.ImageId.unique()[idx]
        img_path = os.path.join(self.dataset_dir, self.image_folder, img_name, "images", img_name + '.png')

        # Load image
        image = Image.open(img_path).convert('RGB')

        return image

    def _load_mask(self, idx): # iloc (int): index in the data frame
        # Collect image name from csv frame
        img_name = self.frame.ImageId.unique()[idx]
        mask_dir = os.path.join(self.dataset_dir, self.image_folder, img_name, "masks")
        mask_paths = [os.path.join(mask_dir, fp) for fp in os.listdir(mask_dir)]
        mask = None
        for fp in mask_paths:
            img = cv2.imread(fp, 0)
            if img is None:
                raise FileNotFoundError("Could not open %s" % fp)
            if mask is None:
                mask = img
            else:
                mask = np.maximum(mask, img)

        mask = Image.fromarray(mask)

        return mask

    def __getitem__(self, idx):
        image = self._get_image(idx)
        mask = self._load_mask(idx)

        # invert if too bright
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, a = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        b = np.count_nonzero(a)
        ttl = np.prod(a.shape)
        if b > ttl / 2:
            image = Image.fromarray(cv2.bitwise_not(img))

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); mask = self.target_transform(mask)
        mask[mask >= 0.5] = 1; mask[mask < 0.5] = 0

        if image.shape[0] == 1: image = image.repeat(3, 1, 1)

        number = len(self.frame[self.frame.ImageId == self.frame.ImageId.unique()[idx]])

        return image, mask

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)