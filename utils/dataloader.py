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
from randaugment import RandAugment #lzc
import random #lzc
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
        # lzc self.transform = transform
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

        self.cls_num=len(self.labels[0])# lzc
        self.class_weight, self.sum_weight = self.get_weight()# lzc
        self.class_dict = self._get_class_dict() #lzc
        ### lzc_start
        self.transform = None
        self.transform_reverse  = transforms.Compose([
                                     transforms.RandomResizedCrop(224,scale =(0.5,1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     RandAugment(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.transform_uniform  =    transforms.Compose([
                                     transforms.RandomResizedCrop(224,scale =(0.5,1.0)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        ### lzc_end

    def get_weight(self):
        max_num = max(self.per_class_labels) #lzc
        class_weight = [max_num / i for i in self.per_class_labels]#lzc
        sum_weight = sum(self.class_weight)#lzc
        return class_weight, sum_weight

    
    def sample_class_index_by_weight(self):#lzc
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def _get_class_dict(self):#lzc
        class_dict = {}  # 初始化类别字典

        for idx, label in enumerate(self.labels):
            label_list = label.tolist()  # 将标签张量转换为列表
            for class_idx, class_label in enumerate(label_list):
                if class_label == 1:  # 判断是否属于该类别
                    if class_idx not in class_dict:
                        class_dict[class_idx] = []
                    class_dict[class_idx].append(idx)  # 将样本索引添加到类别字典中的对应列表中
        return class_dict


    def __len__(self):
        return len(self.labels)

    def get_image_path(self, index):
        return [self.img_path[i] for i in index] 
    
    def get_item(self,index):
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        return sample,label



    def __getitem__(self, index):
        meta_origin = dict()
        meta_balance = dict()
        meta_reverse = dict()
        

        sample,label=self.get_item()
        if self.transform is not None:
            sample = self.transform(sample)
        meta_origin['sample_image'] = sample
        meta_origin['sample_label'] = label
        meta_origin['sample_index'] = index
        #lzc return sample, label, index
        #lzc_start
        

        sample_class = self.sample_class_index_by_weight()
        sample_indexes = self.class_dict[sample_class]
        sample_index = random.choice(sample_indexes)
        sample_reverse,label_reverse=self.get_item(sample_index)
        sample_reverse = self.transform_reverse(sample_reverse)
        meta_reverse['sample_image'] = sample_reverse
        meta_reverse['sample_label'] = label_reverse
        meta_reverse['sample_index'] = sample_index
        
        
        sample_class = random.randint(0, self.cls_num-1)
        sample_indexes = self.class_dict[sample_class]
        sample_index = random.choice(sample_indexes)
        sample_balance,label_balance=self.get_item(sample_index)
        sample_balance = self.transform_uniform(sample_balance)
        meta_balance['sample_image'] = sample_balance
        meta_balance['sample_label'] = label_balance
        meta_balance['sample_index'] = sample_index

        return meta_origin,meta_reverse,meta_balance
        #lzc_end

    


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

    # classes = {i: c["name"] for i, c in enumerate(train_dataset.coco.cats.values())}
    # train_dataset.classes, val_dataset.classes = classes, classes
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
