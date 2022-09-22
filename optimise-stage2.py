import os
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import newdata_utils
import dist_chamfer as ext
from smplx import SMPL, SMPLX
from tqdm import tqdm
from smplx_util import SMPLXProcessor
import pdb


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist

class GMoF_unscaled(nn.Module):
    def __init__(self, rho=1):
        super(GMoF_unscaled, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return dist

def proj_op(points_3d, cam_in, cam_k):
    fx, fy = cam_in[0][0], cam_in[1][1]
    cx, cy = cam_in[0][2], cam_in[1][2]
    # camera distortion coefficients
    k1, k2, p1, p2, k3 = cam_k[0], cam_k[1], cam_k[2], cam_k[3], cam_k[4]
    xp = points_3d[:, 0] / points_3d[:, 2]
    yp = points_3d[:, 1] / points_3d[:, 2]
    r2 = xp.pow(2) + yp.pow(2)
    r4, r6 = r2.pow(2), r2.pow(3)
    xpp = xp*(1+k1*r2+k2*r4+k3*r6) + 2*p1*xp*yp + p2*(r2+2*xp.pow(2))
    ypp = yp*(1+k1*r2+k2*r4+k3*r6) + 2*p2*xp*yp + p1*(r2+2*yp.pow(2))
    u = fx * xpp + cx
    v = fy * ypp + cy
    proj_pnts = torch.cat((u.unsqueeze(1), v.unsqueeze(1)), 1)
    return proj_pnts


def optimise_pipline(frame_num, frame_num_prev, seq_name,
                     human_model, human_model_params, raycast_results,
                     cam_ex, cam_in, cam_k, contacts_all, scene_mesh,
                     joint_mapper, j2ds_openpose,
                     human_params_prev, savepath, init = False):
    if frame_num - frame_num_prev < 7 and not init:
        consec = True
    else:
        consec = False
    reproj_robustifier = GMoF(rho = 100)
    contact_robustifier = GMoF_unscaled(rho = 5e-2)
    human_model = human_model.cuda()
    contacts_all = contacts_all > 0.5
    if raycast_results is not None:
        con_inds_human = raycast_results['vert_inds_human']
        con_loc_scene = torch.Tensor(raycast_results['pnts_loc_scene']).cuda()
        con_inds_scene = raycast_results['face_inds_scene']
        con_norm_scene = torch.Tensor(scene_mesh.face_normals[con_inds_scene]).cuda()
        # only consider non-visible contact points
        contacts_all[con_inds_human] = False
    else:
        con_inds_human, con_loc_scene = None, None
    con_hid_ids = np.nonzero(contacts_all)[0]

    # openpose related
    if j2ds_openpose is not None:
        j2ds_conf = j2ds_openpose[:, 2]
        j2ds_conf_idx = np.argwhere(j2ds_conf > 0.3).reshape(-1)
        j2ds_openpose = torch.Tensor(j2ds_openpose[:, :2]).cuda()
    else:
        j2ds_conf_idx, j2ds_openpose = None, None

    optimise_keys = ['transl', 'global_orient', 'body_pose', 'betas']

    fixed_params = {}
    for key in human_model_params.keys():
        if key not in optimise_keys:
            fixed_params[key] = human_model_params[key].cuda()
    transl = torch.autograd.Variable(human_model_params['transl']).cuda()
    transl.requires_grad = True
    global_orient = torch.autograd.Variable(human_model_params['global_orient']).cuda()
    global_orient.requires_grad = True
    body_pose = torch.autograd.Variable(human_model_params['body_pose']).cuda()
    body_pose.requires_grad = True
    betas = torch.autograd.Variable(human_params_prev['betas']).cuda()
    betas.requires_grad = True
    # params from previous frame
    transl_prev = torch.Tensor(human_params_prev['transl']).cuda()
    global_orient_prev = torch.Tensor(human_params_prev['global_orient']).cuda()
    body_pose_prev = torch.Tensor(human_params_prev['body_pose']).cuda()

    dist_transl = torch.norm((transl-transl_prev).squeeze(), p=2)
    if raycast_results is not None or dist_transl > 0.35:
        # if average original position is too far away from raycast, perform this stage first
        # =============== global translation ================
        print("optimising for contacts...")
        opti_param = [transl]
        optimiser_transl = torch.optim.Adam(opti_param, lr=1e-1)
        iters = 100
        pbar = tqdm(total = iters)
        for i in range(iters):
            out_human = human_model.forward(transl = transl,
                                            global_orient = global_orient,
                                            body_pose = body_pose,
                                            betas = betas,
                                            **fixed_params)
            out_verts = out_human.vertices.squeeze()
            if raycast_results is not None:
                con_verts_human = out_verts[con_inds_human]
                loss_contact = contact_robustifier(con_verts_human - con_loc_scene).mean()
            else:
                loss_contact = 0

            if consec and dist_transl > 0.35:
                loss_tmp_trans = (transl - transl_prev).pow(2).mean()
            else:
                loss_tmp_trans = 0

            loss_transl = loss_contact + 1e-1*loss_tmp_trans

            optimiser_transl.zero_grad()
            try:
                loss_transl.backward()
            except AttributeError:
                break
            optimiser_transl.step()
            pbar.update(1)
            pbar.set_description("loss for translation: {:.4f}"
                                 .format(loss_transl))
        pbar.close()

    # ================ overall ===================
    print("optimising for all...")
    opti_params = [transl, global_orient, body_pose]
    if init:
        opti_params.append(betas)
    optimiser = torch.optim.Adam(opti_params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=100, gamma=0.5)
    max_iter = 100
    pbar = tqdm(total = max_iter)
    for i in range(max_iter):
        out_human = human_model.forward(transl = transl,
                                        global_orient = global_orient,
                                        body_pose = body_pose,
                                        betas = betas,
                                        **fixed_params)
        out_verts = out_human.vertices.squeeze()

        # contact loss, weighted by normal angles
        if raycast_results is not None:
            con_verts_human = out_verts[con_inds_human]
            loss_contact = contact_robustifier(con_verts_human - con_loc_scene).mean()
        else:
            loss_contact = 0

        # openpose reprojection loss
        if j2ds_conf_idx is not None and j2ds_conf_idx.shape[0] > 0:
            j3d_openpose = torch.index_select(out_human.joints, 1, joint_mapper).squeeze()
            j2d = proj_op(j3d_openpose, cam_in, cam_k)
            loss_j2d = reproj_robustifier(j2d[j2ds_conf_idx] - j2ds_openpose[j2ds_conf_idx]).mean()
        else:
            loss_j2d = 0

        # hidden contact loss computed by chamfer loss
        if len(con_hid_ids) == 0:
            loss_contact_hid = 0
        else:
            distChamfer = ext.chamferDist()
            scene_v = torch.tensor(scene_mesh.vertices, dtype=torch.float32).unsqueeze(0).cuda()
            contact_body_hid = out_verts[con_hid_ids].unsqueeze(0)
            contact_dist, _, _, _ = distChamfer(contact_body_hid.contiguous(), scene_v)
            loss_contact_hid = contact_robustifier(contact_dist.sqrt()).mean()

        # temporal smoothness loss
        if consec:
            loss_tmp_trans = (transl - transl_prev).pow(2).mean()
            loss_tmp_ori = ((torch.sin(torch.squeeze(global_orient)) -
                             torch.sin(torch.squeeze(global_orient_prev))).pow(2) +
                            (torch.cos(torch.squeeze(global_orient)) -
                             torch.cos(torch.squeeze(global_orient_prev))).pow(2)).mean()
            loss_tmp_pose = (body_pose - body_pose_prev).pow(2).mean()
        else:
            loss_tmp_trans, loss_tmp_ori, loss_tmp_pose = 0, 0, 0

        loss = 1e+1*loss_contact + 1e+1*loss_contact_hid + 1e-2*loss_j2d\
        + 1e+1*loss_tmp_trans + 1e+2*loss_tmp_ori + 1e+1*loss_tmp_pose

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        pbar.update(1)
        pbar.set_description("loss: {:.4f}".format(loss.item()))
    pbar.close()

    # saving the plys
    # out_human = human_model.forward(transl = transl,
    #                                 global_orient = global_orient,
    #                                 body_pose = body_pose,
    #                                 betas = betas,
    #                                 **fixed_params)
    # out_verts = out_human.vertices.squeeze().detach().cpu().numpy()
    # human_mesh = trimesh.base.Trimesh(vertices=out_verts, faces=human_model.faces, process=False)
    # human_mesh = human_mesh.apply_transform(cam_ex)
    # human_mesh.visual.vertex_colors[con_hid_ids] = [255, 0, 0, 255]

    # save_path = savepath + 'stage2/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # trimesh.exchange.export.export_mesh(human_mesh, os.path.join(save_path,'{:05d}'.format(frame_num)+'.ply'))

    human_model_params['transl'] = transl.detach().cpu()
    human_model_params['global_orient'] = global_orient.detach().cpu()
    human_model_params['body_pose'] = body_pose.detach().cpu()
    human_model_params['betas'] = betas.detach().cpu()

    return human_model_params


if __name__ == '__main__':
    seq_names = ['s1', 's2', 's3', 's4']

    human_model_path = './models/SMPLX_MALE.npz'
    smplxp = SMPLXProcessor(human_model_path)
    joint_mapper = torch.tensor(smplxp.jointmap_openpose(), dtype=torch.long).cuda()

    for seq_name in seq_names:
        savepath = './dataset/'+seq_name+'/'
        print(f'processing {seq_name}')
        data_loader = newdata_utils.load_data(seq_name)
        # avail_frms = data_loader.check_seq_avail(model='RGB')
        avail_path = './dataset/'+seq_name+'/avail_frms.npy'
        avail_frms = np.load(avail_path)
        human_model = data_loader.get_human_model()
        scene_mesh = data_loader.get_scene_mesh()
        world2cam, cam_in, cam_k = data_loader.get_cam_params()
        cam_ex = np.linalg.inv(world2cam)
        scene_mesh = scene_mesh.apply_transform(world2cam)

        save_opt = {}
        for i, frame_num in enumerate(avail_frms):
            print(f'optimising frame {frame_num}...')
            j2ds_openpose = data_loader.load_openpose(frame_num)
            human_model_params = data_loader.get_human_model_params(frame_num, 'RGB')
            contacts = data_loader.get_contacts(frame_num)
            raycast_results = data_loader.get_raycast_results(frame_num)

            if frame_num == avail_frms[0]:
                frame_num_prev = frame_num
                human_params_prev = optimise_pipline(frame_num, frame_num_prev, seq_name,
                                                     human_model, human_model_params,
                                                     raycast_results, cam_ex, cam_in, cam_k,
                                                     contacts, scene_mesh,
                                                     joint_mapper, j2ds_openpose,
                                                     human_model_params, savepath, init=True)
            else:
                frame_num_prev = avail_frms[i-1]
                human_params_prev = optimise_pipline(frame_num, frame_num_prev, seq_name,
                                                     human_model, human_model_params,
                                                     raycast_results, cam_ex, cam_in, cam_k,
                                                     contacts, scene_mesh,
                                                     joint_mapper, j2ds_openpose,
                                                     human_params_prev, savepath)

            save_opt[str(frame_num)] = human_params_prev

        np.save(savepath+'stage2.npy', save_opt)
