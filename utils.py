from PIL import Image
import glob
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

import pdb

class tinydataset(Dataset):
    def __init__(self,folderPath, name_to_index):
        #get the list of all files
        self.file_list = os.listdir(folderPath)
        self.data = []
        for className in self.file_list:
            classPath = os.path.join(folderPath,className)
            img_list = os.listdir(classPath) # get image filename
            for img in img_list:
                imgPath = os.path.join(classPath, img)
                self.data.append((imgPath, name_to_index[className]))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, cls = self.data[idx]
        img = cv2.imread(img_path) #image array with unit8
        img =np.float32(img/255) #rescale pixel value
        img = np.transpose(img, axes = (2,0,1))# channel first
        return img, cls
  

def class_name_to_class_index(folderPath):
    classes = os.listdir(folderPath)
    result = {name:idx for idx, name in enumerate(classes)}
    return result


    