# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:37:08 2019

@author: umaer
"""

import torch
import torch_geometric
from torch_geometric.io import parse_txt_array
from torch_geometric.io.off import face_to_tri
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms.random_flip import RandomFlip
from torch_geometric.transforms.random_rotate import RandomRotate
from sklearn.model_selection import train_test_split
import os
import glob
import random


class Dataset(InMemoryDataset):
    
    def __init__(self, path, train=True, transform=None, pre_transform=None, 
                 pre_filter=None):
        
        self.path = path
        self.train = train
        self.transform=transform
        pt_path = 'C:/tmp/processed/training.pt'
        self.data, self.slices = torch.load(pt_path)

    def read_off(self, file):
        # Some files may contain a bug and do not have a carriage return after OFF.
        with open(file, 'r') as f:
            src = f.read().split('\n')[:-1]
            
            if src[0] == 'OFF':
                src = src[1:]
            else:
                src[0] = src[0][3:]
        
            num_nodes, num_faces = [int(item) for item in src[0].split()[:2]]
        
            pos = parse_txt_array(src[1:1 + num_nodes])
#            pos = pos - pos.mean(dim=-2, keepdim=True)
            scale = (1 / pos.abs().max()) * 0.999999
            pos = pos * scale
        
            face = src[1 + num_nodes:1 + num_nodes + num_faces]
            face = face_to_tri(face)
            
            edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        
            data = Data(edge_index=edge_index, face=face, pos=pos)
    
        return data
    
    def create_dataset(self):
        categories = glob.glob(os.path.join(self.path, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])
        data_list = []
        data_idx = []
        for target, category in enumerate(categories):
                if self.train:
                    folder = os.path.join(self.path, category, 'train')
                else:
                    folder = os.path.join(self.path, category, 'test')
                paths = glob.glob('{}/{}_*.off'.format(folder, category))
                idx = 0
                for path in paths:
                    data = self.read_off(path)
#                    data.y = (target == torch.arange(len(categories)).reshape(1, len(categories))).float()
                    data.y = torch.tensor([target])
                    data_list.append(data)
                    idx += 1
                data_idx.append(idx)
#        return self.collate(data_list)
        return data_list, data_idx
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name, len(self))
    

data_path = 'C:/tmp/raw'
dataset = Dataset(path=data_path, train=True)
dataset = dataset.create_dataset()
dataset, idx = dataset
num_classes, max_class = len(idx), max(idx)

sample_count = 0
for i in range(num_classes):
    class_list = dataset[sample_count:sample_count+idx[i]]
    class_diff = max_class - idx[i]    
    dataset = dataset + random.choices(class_list, k=class_diff)
    sample_count += idx[i]
    
dataset_train, dataset_val = train_test_split(dataset, test_size=0.3, 
                                              random_state=42)

del dataset

rand_rot_x = RandomRotate(20, axis=0)
#rand_rot_y = RandomRotate(20, axis=1)
#rand_rot_z = RandomRotate(20, axis=2)

for i in range(len(dataset_train)):
    dataset_train.append(rand_rot_x(dataset_train[i]))
#    dataset_train.append(rand_rot_y(dataset_train[i]))
#    dataset_train.append(rand_rot_z(dataset_train[i]))
       
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True)


'''
flip = RandomFlip(axis=2)

for i in range(3):
    for i in range(105):
        dataset.append(flip(dataset[i]))
'''
