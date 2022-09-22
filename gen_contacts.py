# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import os.path as osp
import numpy as np
import torch
import trimesh
import sys
from posa_contacts import misc_utils, posa_utils, data_utils
from posa_contacts.cmd_parser import parse_config
from utils import newdata_utils
import pdb

if __name__ == '__main__':
    args, args_dict = parse_config()
    args_dict['batch_size'] = 1
    args_dict['ds_us_dir'] = osp.expandvars(args_dict.get('ds_us_dir'))
    args_dict['model_folder'] = osp.expandvars(args_dict.get('model_folder'))
    args_dict.pop('pkl_file_path')
    seq_names = ['s1', 's2', 's3', 's4']
    for seq_name in seq_names:
        results_path = './dataset/' + seq_name + '/RGB'
        output_dir = './dataset/' + seq_name + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ds_us_dir = args_dict.get('ds_us_dir')

        device = torch.device("cuda" if args_dict.get('use_cuda') else "cpu")
        dtype = torch.float32

        A_1, U_1, D_1 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 1, args_dict['use_cuda'])
        down_sample_fn = posa_utils.ds_us(D_1).to(device)
        up_sample_fn = posa_utils.ds_us(U_1).to(device)

        A_2, U_2, D_2 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 2, args_dict['use_cuda'])
        down_sample_fn2 = posa_utils.ds_us(D_2).to(device)
        up_sample_fn2 = posa_utils.ds_us(U_2).to(device)

        faces_arr = trimesh.load(osp.join(ds_us_dir, 'mesh_{}.obj'.format(0)), process=False).faces

        model = misc_utils.load_model_checkpoint(device=device, **args_dict).to(device)

        data_loader = newdata_utils.load_data(seq_name)
        world2cam, cam_in, cam_k = data_loader.get_cam_params()
        cam_ex = np.linalg.inv(world2cam)
        rotmat = np.zeros((4, 4))
        rotmat[0,0] = -1
        rotmat[1,2] = 1
        rotmat[2,1] = 1
        rotmat[3,3] = 1
        cam_ex = np.dot(rotmat, cam_ex)
        pkl_file_path = os.path.join(results_path, 'results')
        pkl_file_folders = sorted(os.listdir(pkl_file_path))
        pkl_file_paths = [os.path.join(pkl_file_path, fold, '000.pkl') for fold in pkl_file_folders]

        contacts_seq = {}
        for pkl_file_path in pkl_file_paths:
            print('file_name: {}'.format(pkl_file_path))
            if not os.path.isfile(pkl_file_path):
                continue
            frm_num = int(pkl_file_path.split('/')[-2][9:21])
            # load pkl file
            vertices, vertices_can, faces_arr, body_model, R_can, pelvis, torch_param, _ = data_utils.pkl_to_canonical(
                pkl_file_path, cam_ex, device, dtype, **args_dict)

            vertices_can_ds = down_sample_fn.forward(vertices_can.unsqueeze(0).permute(0, 2, 1))
            vertices_can_ds = down_sample_fn2.forward(vertices_can_ds).permute(0, 2, 1).squeeze()

            z = torch.tensor(np.zeros((args.num_rand_samples, args.z_dim)).astype(np.float32)).to(device)
            gen_batch = model.decoder(z, vertices_can_ds.expand(args.num_rand_samples, -1, -1))

            gen_batch = gen_batch.transpose(1, 2)
            gen_batch = up_sample_fn2.forward(gen_batch)
            gen_batch = up_sample_fn.forward(gen_batch)
            gen_batch = gen_batch.transpose(1, 2)

            contacts = gen_batch.view(-1).detach().cpu().numpy()
            contacts_seq[str(frm_num)] = contacts

        np.save(os.path.join(output_dir, 'contacts.npy'), contacts_seq)
        print(f'sequence {seq_name} saved.')
