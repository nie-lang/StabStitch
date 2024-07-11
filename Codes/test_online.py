# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from spatial_network import build_SpatialNet, SpatialNet
from temporal_network import build_TemporalNet, TemporalNet
from smooth_network import build_SmoothNet, SmoothNet
import os
import numpy as np
import skimage
import cv2
import utils.torch_tps_transform as torch_tps_transform
import utils.torch_tps_transform_point as torch_tps_transform_point
from PIL import Image
import glob
import time
from torchvision.transforms import GaussianBlur



last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')

grid_h = 6
grid_w = 8

def linear_blender(ref, tgt, ref_m, tgt_m, mask=False):
    blur = GaussianBlur(kernel_size=(21,21), sigma=20)
    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)

    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())

    vec = (center2[0] - center1[0], center2[1] - center1[1])

    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
    r, c = torch.nonzero(ovl[0, 0], as_tuple=True)

    ovl_mask = torch.zeros_like(ref_m_).cuda()
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / (proj_val.max() - proj_val.min() + 1e-3)

    mask1 = (blur(ref_m_ + (1-ovl_mask)*ref_m[:,0].unsqueeze(1)) * ref_m + ref_m_).clamp(0,1)
    if mask: return mask1

    mask2 = (1-mask1) * tgt_m
    stit = ref * mask1 + tgt * mask2

    return stit


def recover_mesh(norm_mesh, height, width):
    #from [bs, pn, 2] to [bs, grid_h+1, grid_w+1, 2]

    batch_size = norm_mesh.size()[0]
    mesh_w = (norm_mesh[...,0]+1) * float(width) / 2.
    mesh_h = (norm_mesh[...,1]+1) * float(height) / 2.
    mesh = torch.stack([mesh_w, mesh_h], 2) # [bs,(grid_h+1)*(grid_w+1),2]

    return mesh.reshape([batch_size, grid_h+1, grid_w+1, 2])

def get_rigid_mesh(batch_size, height, width):


    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2


# bs, T, h, w, 2  smooth_path
def get_stable_sqe(img1_list, img2_list, ori_mesh):
    batch_size, _, img_h, img_w = img2_list[0].shape

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)

    width_max = torch.max(torch.max(ori_mesh[...,0]))
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(ori_mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(ori_mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(ori_mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    width_min = width_min.int()
    width_max = width_max.int()
    height_min = height_min.int()
    height_max = height_max.int()
    out_width = width_max - width_min+1
    out_height = height_max - height_min+1

    print(out_width)
    print(out_height)

    img1_warp = torch.zeros((1, 3, out_height, out_width)).cuda()

    stable_list = []
    mesh_tran_list = []
    for i in range(len(img2_list)):
        mesh = ori_mesh[:,i,:,:,:]
        mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
        norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
        img2 = (img2_list[i].cuda()+1)*127.5

        # mode = 'FAST': use F.grid_sample to interpolate. It's fast, but may produce thin black boundary.
        # mode = 'NORMAL': use our implemented interpolation function. It's a bit slower, but avoid the black boundary.
        img2_warp = torch_tps_transform.transformer(img2, norm_mesh, norm_rigid_mesh, (out_height, out_width), mode = 'FAST')

        # average blending
        img1_warp[:,:,int(0-height_min):int(0-height_min)+360, int(0-width_min):int(0-width_min)+480] = (img1_list[i].cuda() + 1) * 127.5

        ave_fusion = (img1_warp[0] * (img1_warp[0]/ (img1_warp[0]+img2_warp[0]+1e-6)) + img2_warp[0] * (img2_warp[0]/ (img1_warp[0]+img2_warp[0]+1e-6)))

        # # linear blending
        # img1_warp[:,int(0-height_min):int(0-height_min)+360, int(0-width_min):int(0-width_min)+480] = (img1_list[i][0] + 1) * 127.5
        # mask1 = img1_warp.clone()
        # mask1[:,int(0-height_min):int(0-height_min)+360, int(0-width_min):int(0-width_min)+480] = 1
        # mask2 = torch_tps_transform.transformer(torch.ones_like(img2).cuda(), norm_mesh, norm_rigid_mesh, (out_height, out_width))
        # ave_fusion = linear_blender(img1_warp.unsqueeze(0), img2_warp, mask1.unsqueeze(0), mask2)
        # ave_fusion = ave_fusion[0]

        stable_list.append(ave_fusion)


    return stable_list, out_width, out_height



def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # define the network
    spatial_net = SpatialNet()
    temporal_net = TemporalNet()
    smooth_net = SmoothNet()
    if torch.cuda.is_available():
        spatial_net = spatial_net.cuda()
        temporal_net = temporal_net.cuda()
        smooth_net = smooth_net.cuda()

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()

    if len(ckpt_list) == 3:
        # load spatial warp model
        spatial_model_path = MODEL_DIR + "/spatial_warp.pth"
        spatial_checkpoint = torch.load(spatial_model_path)
        spatial_net.load_state_dict(spatial_checkpoint['model'])
        print('load model from {}!'.format(spatial_model_path))
        # load temporal warp model
        temporal_model_path = MODEL_DIR + "/temporal_warp.pth"
        temporal_checkpoint = torch.load(temporal_model_path)
        temporal_net.load_state_dict(temporal_checkpoint['model'])
        print('load model from {}!'.format(temporal_model_path))
        # load smooth warp model
        smooth_model_path = MODEL_DIR + "/smooth_warp.pth"
        smooth_checkpoint = torch.load(smooth_model_path)
        smooth_net.load_state_dict(smooth_checkpoint['model'])
        print('load model from {}!'.format(smooth_model_path))
    else:
        print('No checkpoint found!')



    spatial_net.eval()
    temporal_net.eval()
    smooth_net.eval()

    print("##################start testing#######################")

    psnr_list = []
    ssim_list = []


    video_name_list = glob.glob(os.path.join(args.test_path, '*'))
    video_name_list = sorted(video_name_list)
    print(video_name_list)

    img_h = 360
    img_w = 480



    for i in range(len(video_name_list)):
        # if i<24:
        #     continue
        print()
        print(i)
        print(video_name_list[i])

        #define an online buffer (len == 7)
        buffer_len = 7
        tmotion_tensor_list = []
        smotion_tensor_list = []
        omask_tensor_list = []

        # img name list
        img1_name_list = glob.glob(os.path.join(video_name_list[i]+ "/video1/", '*.jpg'))
        img2_name_list = glob.glob(os.path.join(video_name_list[i]+ "/video2/", '*.jpg'))
        img1_name_list = sorted(img1_name_list)
        img2_name_list = sorted(img2_name_list)


        # prepare folders
        if not os.path.exists('../result/'):
            os.makedirs('../result/')
        video_name = video_name_list[i].split('/')[-1] + ".mp4"
        media_path = '../result/' + video_name
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 30

        img1_tensor_list = []
        img2_tensor_list = []

        # load imgs
        for k in range(0, len(img2_name_list)):
            img1 = cv2.imread(img1_name_list[k])
            img1 = cv2.resize(img1, (img_w, img_h))
            img1 = img1.astype(dtype=np.float32)
            img1 = np.transpose(img1, [2, 0, 1])
            img1 = (img1 / 127.5) - 1.0
            img1_tensor = torch.tensor(img1).unsqueeze(0)#.cuda()
            img1_tensor_list.append(img1_tensor)

            img2 = cv2.imread(img2_name_list[k])
            img2 = cv2.resize(img2, (img_w, img_h))
            img2 = img2.astype(dtype=np.float32)
            img2 = np.transpose(img2, [2, 0, 1])
            img2 = (img2 / 127.5) - 1.0
            img2_tensor = torch.tensor(img2).unsqueeze(0)#.cuda()
            img2_tensor_list.append(img2_tensor)



        start_time1 = time.time()
        NOF = len(img2_name_list)
        # motion estimation
        for k in range(0, len(img2_name_list)):

            # step 1: spatial warp
            with torch.no_grad():
                spatial_batch_out = build_SpatialNet(spatial_net, img1_tensor_list[k].cuda(), img2_tensor_list[k].cuda())
            smotion = spatial_batch_out['motion']
            omask = spatial_batch_out['overlap_mesh']

            smotion_tensor_list.append(smotion)
            omask_tensor_list.append(omask)


        # step 2: temporal warp
        with torch.no_grad():
            temporal_batch_out = build_TemporalNet(temporal_net, img2_tensor_list)
        tmotion_tensor_list = temporal_batch_out['motion_list']


        print("fps (spatial & temporal warp):")
        print(NOF/(time.time() - start_time1))


        ##############################################
        #############   data preparation  ############
        # converting tmotion (t-th frame) into tsmotion ( (t-1)-th frame )
        rigid_mesh = get_rigid_mesh(1, img_h, img_w)
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
        smesh_list = []
        tsmotion_list = []
        for k in range(len(tmotion_tensor_list)):
            smotion = smotion_tensor_list[k]
            smesh = rigid_mesh + smotion
            if k == 0:
                tsmotion = smotion.clone() * 0

            else:
                smotion_1 = smotion_tensor_list[k-1]
                smesh_1 = rigid_mesh + smotion_1
                tmotion = tmotion_tensor_list[k]
                tmesh = rigid_mesh + tmotion
                norm_smesh_1 = get_norm_mesh(smesh_1, img_h, img_w)
                norm_tmesh = get_norm_mesh(tmesh, img_h, img_w)
                tsmesh = torch_tps_transform_point.transformer(norm_tmesh, norm_rigid_mesh, norm_smesh_1)
                tsmotion = recover_mesh(tsmesh, img_h, img_w) - smesh
            # append
            smesh_list.append(smesh)
            tsmotion_list.append(tsmotion)


        # step 3: smooth warp
        ori_mesh = 0
        target_mesh = 0


        for k in range(len(tmotion_tensor_list)-6):
            # get sublist and set the first element to 0
            tsmotion_sublist = tsmotion_list[k:k+7]
            tsmotion_sublist[0] = smotion_tensor_list[k] * 0


            with torch.no_grad():
                smooth_batch_out = build_SmoothNet(smooth_net, tsmotion_sublist, smesh_list[k:k+7], omask_tensor_list[k:k+7])

            _ori_mesh = smooth_batch_out["ori_mesh"]
            _target_mesh = smooth_batch_out["target_mesh"]


            if k == 0:
                ori_mesh = _ori_mesh
                target_mesh = _target_mesh

            else:
                ori_mesh = torch.cat((ori_mesh, _ori_mesh[:,-1,...].unsqueeze(1)), 1)
                target_mesh = torch.cat((target_mesh, _target_mesh[:,-1,...].unsqueeze(1)), 1)




        print("fps (smooth warp):")
        print(NOF/(time.time() - start_time1))

        # # warp with the original mesh
        # stable_list, out_width, out_height = get_stable_sqe(img1_tensor_list, img2_tensor_list, ori_mesh)
        # warp with the target mesh
        stable_list, out_width, out_height = get_stable_sqe(img1_tensor_list, img2_tensor_list, target_mesh)


        print("fps (warping & average blending):")
        print(NOF/(time.time() - start_time1))



        media_writer = cv2.VideoWriter(media_path, fourcc, fps, (out_width, out_height))

        for k in range(len(stable_list)):

            ave_fusion = stable_list[k].cpu().numpy().transpose(1,2,0)
            media_writer.write(ave_fusion.astype(np.uint8 ))

        media_writer.release()





    print("##################end testing#######################")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='/opt/data/StabStitch-D/testing/')
    #parser.add_argument('--test_path', type=str, default='/opt/data/Tra-Dataset2/')


    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)