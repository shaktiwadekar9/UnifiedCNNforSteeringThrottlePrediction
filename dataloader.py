"""
@author: ShaktiWadekar
"""

import torch
import numpy as np
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image

from scipy.ndimage import gaussian_filter1d


class SimulationDataset(Dataset):
    """Dataset wrapping input and target tensors for the driving simulation dataset.

    Arguments:
        mode (String):  Dataset - train, test
        path (String): Path to the csv file with the image paths and the target values
    """

    def __init__(self, mode, csv_path, steering_or_throttle, transforms=None):

        self.transforms = transforms

        self.data = pd.read_csv(csv_path, header=None)

        # First column contains the middle image paths
        # Fourth column contains the steering angle
        # Fifth column contains the throttle angle
        start = 0
        end = int(4 / 5 * len(self.data))

        if (mode == "test"):
            start = end 
            end = len(self.data)

        self.image_paths = np.array(self.data.iloc[start:end, 0:3])
        
        if steering_or_throttle == "steering":
            self.targets = np.array(self.data.iloc[start:end, 3])
        elif steering_or_throttle == "throttle": 
            self.targets = np.array(self.data.iloc[start:end, 4])
        else:
            raise ValueError('steering_or_throttle variable must be either \
                             "steering" or "throttle"' )
            

        # Preprocess and filter data
        self.targets = gaussian_filter1d(self.targets, 2)      
        
        #bias = 0.03  ##change for both
        self.image_paths = [image_path for image_path, target in zip(self.image_paths, self.targets)]# if abs(target) > bias]  ##change for both
        self.targets = [target for target in self.targets]# if abs(target) > bias]  ##change for both

    def __getitem__(self, index):

         # Get image name from the pandas df
        image_paths = self.image_paths[index]
        # Open image
        image = [Image.open(image_paths[i]) for i in range(3)]
        target = self.targets[index]     

        sample = {'image': image, 'target': target}

        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample['image'], sample['target']

    def __len__(self):
        return len(self.image_paths)
