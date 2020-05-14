import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
#from utils import *

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]
           
class ShapeNet(data.Dataset): 
    def __init__(self, train = True, npoints = 8192):
        if train:
            self.list_path = './data/train.list'
        else:
            self.list_path = './data/val.list'
        self.npoints = npoints
        self.train = train

        with open(os.path.join(self.list_path)) as file:
            self.model_list = [line.strip().replace('/', '_') for line in file]
        random.shuffle(self.model_list)
        self.len = len(self.model_list * 50)

    def __getitem__(self, index):
        model_id = self.model_list[index // 50]
        scan_id = index % 50
        def read_pcd(filename):
            pcd = o3d.io.read_point_cloud(filename)
            return torch.from_numpy(np.array(pcd.points)).float()
        if self.train:
            partial = read_pcd(os.path.join("./data/train/", model_id + '_%d_denoised.pcd' % scan_id))
        else:
            partial = read_pcd(os.path.join("./data/val/", model_id + '_%d_denoised.pcd' % scan_id))
        complete = read_pcd(os.path.join("./data/complete/", '%s.pcd' % model_id))     
        __import__('pdb').set_trace()
        return model_id, resample_pcd(partial, 5000), resample_pcd(complete, self.npoints)  # normalized pointclouds

    def __len__(self):
        return self.len

class PCN(data.Dataset):  # for evaluation(test), not for training
    def __init__(self, train = True, npoints = None):
        if train:
            self.list_path = './data/pcn_shapenet/train.txt'
            self.data_root = './data/pcn_shapenet/train'
        else:
            self.list_path = './data/pcn_shapenet/val.txt'
            self.data_root = './data/pcn_shapenet/val'
        self.npoints = npoints
        self.train = train

        with open(self.list_path, 'r') as f:
            # list of voxel data files in voxel
            self.model_list = [line[:-1] for line in f]  # remove '\n'

        random.shuffle(self.model_list)
        self.len = len(self.model_list)

    def __getitem__(self, index):
        model_id = self.model_list[index]
        # scan_id = index % 50
        # def read_pcd(filename):
        #     pcd = o3d.io.read_point_cloud(filename)
        #     return torch.from_numpy(np.array(pcd.points)).float()
        # if self.train:
        #     partial = read_pcd(os.path.join("./data/train/", model_id + '_%d_denoised.pcd' % scan_id))
        # else:
        #     partial = read_pcd(os.path.join("./data/val/", model_id + '_%d_denoised.pcd' % scan_id))
        # complete = read_pcd(os.path.join("./data/complete/", '%s.pcd' % model_id))       
        data_path = os.path.join(self.data_root, model_id)
        input_path = data_path + '_input.pt'
        gt_path = data_path +'_gt.pt'
        voxel_size = 1/6

        # data saved as sparse tensor...
		x_coord = torch.load(input_path, map_location=self.device) #* voxel_size - 0.5
		# x_coord = torch.unique(x_coord, dim=0)
		# x_feat = torch.ones(x_coord.shape[0], 1)
		y_coord = torch.load(gt_path, map_location=self.device) #* voxel_size - 0.5
		# y_coord = torch.unique(y_coord, dim=0)
		# y_feat = torch.ones(y_coord.shape[0], 1)

        
        return model_id, resample_pcd(partial, 5000), resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len
