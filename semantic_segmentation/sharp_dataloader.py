# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
from tqdm import tqdm
# from osgeo import gdal
import rasterio
from os.path import dirname as up
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import pytorch_lightning as pl
import pandas as pd


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

###############################################################
# Pixel-level Semantic Segmentation Data Loader               #
###############################################################

class GenMARSH(Dataset):
    
    def __init__(self, csv_file, normalization=None, transform=None, ndvi=False, ndwi=False, datasource=None, label_agg=True, high_low=True, binary_class=False):
        
        
        self.df = pd.read_csv(csv_file)
        self.normalization = normalization
        self.ndvi = ndvi
        self.ndwi = ndwi
        self.transform = transform
        self.datasource = datasource
        self.label_agg = label_agg
        self.high_low = high_low
        self.binary_class = binary_class
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx, 1]
        
        # the dimension is CxHxW
        image = rasterio.open(img_name).read().astype('float32') 
    
        target = self.df.iloc[idx, 2]
        target = rasterio.open(target).read().astype('int8') # 1xHxW
#         target = rasterio.open(target).read(1).astype('int8') # 1xHxW
        
#         nan_mask = np.isnan(image)
#         image[nan_mask] = self.impute_nan[nan_mask]
        
        
        if self.label_agg:
            target[target==9]=8 # z7 has an unknown class 9, merge it with class 8-upland
        
        if self.high_low:
            target[target==3]=0
            target[target==4]=0
            target[target==5]=0
            target[target==6]=0
            target[target==7]=0
            target[target==8]=0
        
        if self.binary_class:
            target[target==2]=1

        if self.normalization:
            if self.datasource == 'naip':
                image = image/255. # image = image/10000*3.5 (sentinel)
            elif self.datasource == 'sentinel':
                image = image/10000*3.5
            else:
                raise("The data source should be NAIP or sentinel")
        
        # the image bands in the input data are: B2, B3, B4, B8, B1, ...., B12 (Sentinel: RGBNIR + ....)
        if self.ndvi:
            ndvi = (image[3,:,:] - image[2,:,:]) / (image[3,:,:] + image[2,:,:]) # (NIR - R) / (NIR + R)    
            ndvi = ndvi[np.newaxis, :, :]
            image = np.concatenate([image, ndvi], axis=0).astype('float32')
        
        if self.ndwi and self.datasource.lower()=='sentinel':
            ndwi = (image[1,:,:] - image[3,:,:]) / (image[1,:,:] + image[3,:,:]) # NDWI = (G-NIR)/(G+NIR)
            ndwi = ndwi[np.newaxis, :, :]
            image = np.concatenate([image, ndwi], axis=0).astype('float32')
        
        # Need to double check here!!!!!
        if self.transform is not None:
            target = target[np.newaxis,:,:]
            image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]).astype('float32') # CxHxW to HxWxC
            target = np.moveaxis(target, [0, 1, 2], [2, 0, 1])    # CxWxH to WxHxC
            stack = np.concatenate([image, target], axis=-1).astype('float32') # In order to rotate-transform both mask and image
            stack = self.transform(stack)
            
            image = stack[:-1,:,:]
            target = stack[-1,:,:].long()[0] # Recast target values back to int64 or torch long dtype
        
#         image = image[[0,1,2,3], :, :] # Use only four bands from Sentinel for testing.
        sample = {'image': image, 'label': target}
        
        return sample


###############################################################
# Transformations                                             #
###############################################################
class RandomRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)

class Resize:
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size
    
    def __call__(self, img: np.ndarray):
        
        return F.resize(img, self._size)

# class NAIPnormalization:
#     def __init__(self):
    
#     def __call__(self, x):
#         normalizaed = x / 255.
#         return normalized
    

