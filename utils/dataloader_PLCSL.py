"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
import json
from PIL import Image
from utils import  ClassAwareSampler

# Image statistics
RGB_statistics = {
    'MLT_coco': {
        # 'mean': [123.675, 116.28, 103.53],
        # 'std': [58.395, 57.12, 57.375]
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'MLT_voc': {
        'mean': [0.485, 0.456,  0.406],
        'std':[0.229, 0.224, 0.225]
        # 'mean': [123.675, 116.28, 103.53],
        # 'std':[58.395, 57.12, 57.375]
    }
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, data_path, annatation_path, transform):
        temp=0
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.index_number=dict()
        with open(annatation_path) as f:
            row_data = json.load(f)  
            for key in row_data.keys():
                self.img_path.append(data_path+key)
                self.labels.append(
                    torch.from_numpy(np.array(row_data[key]))    
                    )
                if row_data[key][0]==1:
                    temp+=1    
        self.per_class_labels= np.sum(self.labels, axis = 0)
        self.number=len(self.img_path)
        # low=0
        # many=0
        # medium=0
        # print (self.per_class_labels)
        # for number in self.per_class_labels:
        #     if number>300:
        #         many+=1
        #     elif number <50:
        #         low+=1
        #     else:
        #         medium+=1
        # print (many)
        # print (medium)
        # print (low)
        
    def __len__(self):
        return len(self.labels)

    def get_image_path(self, index):
        return [self.img_path[i] for i in index] 
        
    def __getitem__(self, index):
        # print (index)
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

    


# Load datasets
def load_data(dataloader_setting, sampler_dic=None):


    train_annatation_path=dataloader_setting["train_annatation_path"]
    val_annatation_path=dataloader_setting["val_annatation_path"]
    train_data_path=dataloader_setting["train_data_path"]
    val_data_path=dataloader_setting["val_data_path"]

    key = dataloader_setting['dataset']
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
    train_transform = get_data_transform('train', rgb_mean, rgb_std, key)
    print('Use data transformation:', train_transform)
    val_transform = get_data_transform('val', rgb_mean, rgb_std, key)
    
    
    train_dataset = LT_Dataset(train_data_path, train_annatation_path, train_transform)
    print('Load train set, length is '+ str(len(train_dataset.img_path))+".")
    val_dataset = LT_Dataset(val_data_path, val_annatation_path, val_transform)
    
    # print('=====> Using sampler: ', sampler_dic['sampler'])
    # print('=====> Sampler parameters: ', sampler_dic['params'])
    # sampler = ClassAwareSampler(data_source=train_dataset, reduce=4)
    # sampler=ClassAwareSampler(data_source=train_dataset)
    # train_loader=DataLoader(dataset=train_dataset, batch_size=dataloader_setting['batch_size'], shuffle=False, sampler=sampler,
    #                        num_workers=dataloader_setting['num_workers'])
    train_loader=DataLoader(dataset=train_dataset, batch_size=dataloader_setting['batch_size'], shuffle=False, sampler=sampler_dic,
                           num_workers=dataloader_setting['num_workers'])
    val_loader=DataLoader(dataset=val_dataset, batch_size=dataloader_setting['batch_size'],
                          shuffle=True, num_workers=dataloader_setting['num_workers'])

    return train_loader, val_loader
