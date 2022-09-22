import os
import numpy as np
import torch
import torch.nn as nn
import scipy
import scipy.sparse

import json
import pickle
import sys
import trimesh
from smplx import SMPL, SMPLX
import pdb

class load_data():
    def __init__(self, seq_name):
        self.seq_name = seq_name
        # set the paths
        self.data_path = './dataset/'
        self.scene_name = 'scene'
        self.frm_start = 9
        self.frm_end = 21
        self.human_model_path = './models/SMPLX_MALE.npz'

    def check_seq_avail(self, model = 'RGB'):
        pkls_path = os.path.join(self.data_path, self.seq_name, model, 'results')
        pkl_list = os.listdir(pkls_path)
        pkl_list = [pkl for pkl in pkl_list if os.path.isfile(os.path.join(pkls_path, pkl, '000.pkl'))]
        frame_nums = [int(fold[self.frm_start:self.frm_end]) for fold in pkl_list]
        return sorted(frame_nums)

    def get_human_model(self):
        human_model = SMPLX(self.human_model_path, num_pca_comps=12, create_global_orient=True)
        return human_model

    def get_human_model_params(self, frame_num, model = 'RGB'):
        # read the smplx parameters from the pkl file and the human models
        # model can be 'PROXD', 'PROX', 'RGB', 'SMPLifyD'
        pkls_path = os.path.join(self.data_path, self.seq_name, model, 'results')
        pkl_list = os.listdir(pkls_path)
        pkl_foldername = [fold for fold in pkl_list if int(fold[self.frm_start:self.frm_end]) == frame_num][0]
        body_params = pickle.load(open(os.path.join(pkls_path, pkl_foldername, '000.pkl'), 'rb'))
        ignore_params = ['pose_embedding', 'camera_rotation', 'camera_translation',
                         'jaw_pose', 'leye_pose', 'reye_pose', 'expression',
                         'num_pca_comps']
        # params are in camera coordinates already
        human_model_params = {}
        for key in body_params.keys():
            if key in ignore_params:
                continue
            else:
                human_model_params[key] = torch.Tensor(body_params[key])
        return human_model_params

    def get_smplx_gt(self, frame_num):
        # world coordinate
        npy_path = os.path.join(self.data_path, self.seq_name, 'smplx.npy')
        smplx_gt = np.load(npy_path, allow_pickle=True).item()
        smplx_gt_frm = smplx_gt[str(frame_num)]
        human_model_params = {}
        for key in smplx_gt_frm.keys():
            human_model_params[key] = torch.Tensor(smplx_gt_frm[key])
        return human_model_params

    def load_openpose(self, frm):
        json_path = os.path.join(self.data_path, self.seq_name, 'keypoints')
        json_list = os.listdir(json_path)
        json_name = [js for js in json_list if int(js[self.frm_start:self.frm_end]) == frm][0]
        json_path = os.path.join(json_path, json_name)
        with open(json_path) as keypoint_file:
            data = json.load(keypoint_file)
        if data['people'] == []:
            return None
        person_data = data['people'][0]
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                   dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        return body_keypoints

    def get_image_path(self, frame_num, folder='stream'):
        img_path = os.path.join(self.data_path, self.seq_name, folder)
        img_names = os.listdir(img_path)
        img_name = [name for name in img_names if int(name.split('.')[0])-1 == frame_num][0]
        img_path_final = os.path.join(img_path, img_name)
        return img_path_final

    def get_scene_mesh(self):
        # scene mesh (world coordinate)
        scene_path = os.path.join(self.data_path, self.seq_name, 'scene', self.scene_name + '.ply')
        scene_mesh = trimesh.load(scene_path, process = False)
        return scene_mesh

    def get_cam_params(self):
        # camera params
        cam_path = os.path.join(self.data_path, self.seq_name, 'cam.json')
        with open(cam_path, 'r') as camera:
            cam_params = json.load(camera)
            world2cam = np.asarray(cam_params['camera_extrinsics'])
            world2cam[:3, 3] /= 1000
            cam_in = np.asarray(cam_params['camera_intrinsics'])
            cam_k = np.asarray(cam_params['k'])
        return world2cam, cam_in, cam_k

    def get_contacts(self, frame_num):
        contacts_path = os.path.join(self.data_path, self.seq_name, 'contacts.npy')
        contacts_results = np.load(contacts_path, allow_pickle=True).item()
        contact_frm = contacts_results[str(frame_num)]
        return contact_frm

    def get_raycast_results(self, frame_num):
        raycast_results_path = os.path.join(self.data_path, self.seq_name, 'raycast.npy')
        raycast_results = np.load(raycast_results_path, allow_pickle = True).item()
        result_frm = raycast_results[str(frame_num)]
        return result_frm

    def get_mask(self, frame_num):
        mask_path = os.path.join(self.data_path, self.seq_name, 'masks.npy')
        mask_result = np.load(mask_path, allow_pickle = True).item()
        mask_frm = mask_result[str(frame_num)]
        return mask_frm

    def load_sdf(self):
        device = torch.device('cuda')
        dtype = torch.float32
        sdf_dir = os.path.join(self.data_path, self.seq_name, 'scene')
        with open(os.path.join(sdf_dir, self.scene_name + '.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_min = torch.tensor(np.array(sdf_data['min']), dtype=dtype, device=device)
            grid_max = torch.tensor(np.array(sdf_data['max']), dtype=dtype, device=device)
            grid_dim = sdf_data['dim']
        voxel_size = (grid_max - grid_min) / grid_dim
        sdf = np.load(os.path.join(sdf_dir, self.scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
        sdf = torch.tensor(sdf, dtype=dtype, device=device)
        return sdf, grid_max, grid_min, voxel_size
