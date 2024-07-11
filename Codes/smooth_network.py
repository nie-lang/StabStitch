import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import utils.torch_tps_transform_point as torch_tps_transform_point
import ssl
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as T
import random
from torch import nn, einsum
from einops import rearrange


grid_h = 6
grid_w = 8




def build_SmoothNet(net, tsmotion_list, smesh_list, omask_tensor_list):

    # predict the delta motion for tsflow
    stitch_mesh, overlap_mask, ori_path, delta_motion = net(smesh_list, omask_tensor_list, tsmotion_list)

    # get the smoothes tsflow
    smooth_path = ori_path + delta_motion
    # get the actual warping mesh
    target_mesh = stitch_mesh - delta_motion  # bs, T, h, w, 2

    out_dict = {}
    out_dict.update(ori_path = ori_path, smooth_path = smooth_path, ori_mesh = stitch_mesh, target_mesh = target_mesh, overlap_mask = overlap_mask, delta_motion = delta_motion)

    return out_dict



# define and forward ( Because of the load is unbalanced when use torch.nn.DataParallel, we define warp in forward)
class SmoothNet(nn.Module):

    def __init__(self, dropout=0.):
        super(SmoothNet, self).__init__()


        self.MotionPre = MotionPre1()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # forward
    def forward(self, smesh_list, omask_list, tsmotion_list):

        # to generate meshflow from the first motion
        tsflow_list = [tsmotion_list[0]]
        for i in range(1, len(tsmotion_list)):
            tsflow_list.append(tsflow_list[i-1] + tsmotion_list[i])


        # generate meshflow tensor
        smesh = torch.cat(smesh_list, 3)  # bs, h, w, 2*T
        omask = torch.cat(omask_list, 3)  # bs, h, w, 2*T
        tsflow = torch.cat(tsflow_list, 3)  # bs, h, w, 2*T

        smesh = smesh.reshape(-1, grid_h+1, grid_w+1, len(smesh_list), 2)     # bs, h, w, T, 2
        smesh = smesh.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        omask = omask.reshape(-1, grid_h+1, grid_w+1, len(omask_list), 1)     # bs, h, w, T, 2
        omask = omask.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2

        tsflow = tsflow.reshape(-1, grid_h+1, grid_w+1, len(tsflow_list), 2)     # bs, h, w, T, 2
        tsflow = tsflow.permute(0, 3, 1, 2, 4)   # bs, T, h, w, 2



        delta_tsflow = self.MotionPre(smesh, omask, tsflow)


        return smesh, omask, tsflow, delta_tsflow



class MotionPre1(nn.Module):
    def __init__(self, kernel = 5):
        super().__init__()

        self.embedding1 = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU()
        )
        self.embedding2 = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU()
        )
        self.embedding3 = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        self.pad = kernel // 2
        self.MotionConv3D = nn.Sequential(
            nn.Conv3d(64, 64, (kernel, 3, 3), padding=(self.pad, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 64, (kernel, 3, 3), padding=(self.pad, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 64, (kernel, 3, 3), padding=(self.pad, 1, 1)),
            nn.ReLU()
        )

        self.decoding = nn.Sequential(
            nn.Linear(64, 2)
        )



    def forward(self, smesh, omask, tsflow):
        # input: meshflow -- bs, T, h, w, 2
        # output: delta_meshflow -- bs, T, h, w, 2

        # predict the delta motion to stabilize the last meshflow
        hidden1 = self.embedding1(smesh)       # bs, T, H, W, 32
        hidden2 = self.embedding2(omask)       # bs, T, H, W, 32
        hidden3 = self.embedding3(tsflow)       # bs, T, H, W, 32

        hidden = torch.cat([hidden1, hidden2, hidden3], 4)    # bs, T, H, W, 64
        hidden = self.MotionConv3D(hidden.permute(0, 4, 1, 2, 3))   # bs, 64, T, H, W
        delta_tsflow = self.decoding(hidden.permute(0, 2, 3, 4, 1))   # bs, T, H, W, 2


        return delta_tsflow
