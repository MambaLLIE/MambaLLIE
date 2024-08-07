import os.path as osp
import torch
import torch.utils.data as data
import basicsr.data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os
import functools
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Dataset_SIDImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_SIDImage, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.io_backend_opt = opt['io_backend']
        self.data_type = opt['io_backend']
        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}

        subfolders_LQ_origin = util.glob_file_list(self.LQ_root)
        subfolders_GT_origin = util.glob_file_list(self.GT_root)
        subfolders_LQ = []
        subfolders_GT = []
        if self.opt['phase'] == 'train':
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '0' in name[0] or '2' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
        else:
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '1' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])

        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            # for frames in each video:
            subfolder_name = osp.basename(subfolder_LQ)

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)

            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(
                img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

            border_l = [0] * max_idx
            for i in range(self.half_N_frames):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            if self.cache_data:
                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        img_LQ_path = self.imgs_LQ[folder][idx]
        img_LQ_path = [img_LQ_path]
        img_GT_path = self.imgs_GT[folder][0]
        img_GT_path = [img_GT_path]
        
        if self.opt['phase'] == 'train':
            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            rlt = util.augment_torch(
                img_LQ_l, self.opt['use_hflip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]

            _, H, W = img_GT.shape  # real img size
            gt_size = self.opt['gt_size']
            rnd_h = random.randint(0, max(0, H - gt_size))
            rnd_w = random.randint(0, max(0, W - gt_size))
            img_LQ = img_LQ[:, rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size]
            img_GT = img_GT[:, rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size]


        elif self.opt['phase'] == 'test':
            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

        else:
            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

        return {
            'lq': img_LQ,
            'gt': img_GT,
            # 'nf': img_nf,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': img_LQ_path[0],
            'gt_path': img_GT_path[0]
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])