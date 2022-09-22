import os
import sys
import numpy as np
import open3d as o3d
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import newdata_utils
from utils import dist_chamfer as ext
from utils.misc import *
from utils.smplx_util import SMPLXProcessor
from tqdm import tqdm
import pdb


def optimise_pipline(frame_num, frame_num_prev, seq_name,
                     human_model, human_model_params, raycast_results,
                     cam_ex, cam_in, cam_k, contacts_all, scene_mesh,
                     joint_mapper, j2ds_openpose,
                     human_params_prev, savepath, init = False):
    if frame_num - frame_num_prev < 4 and not init:
        consec = True
    else:
        consec = False
    reproj_robustifier = GMoF(rho = 100)
    contact_robustifier = GMoF_unscaled(rho = 5e-2)
    human_model = human_model.cuda()
    contacts_all = contacts_all > 0.5
    con_all_inds = np.nonzero(contacts_all)[0]
    if raycast_results is not None:
        con_inds_human = raycast_results['vert_inds_human']
        con_loc_scene = torch.Tensor(raycast_results['pnts_loc_scene']).cuda()
        con_inds_scene = raycast_results['face_inds_scene']
        con_norm_scene = torch.Tensor(scene_mesh.face_normals[con_inds_scene]).cuda()
        # only consider non-visible contact points
        contacts_all[con_inds_human] = False
        scene_con_verts = scene_mesh.faces[con_inds_scene].reshape(-1)
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
    betas = torch.autograd.Variable(human_model_params['betas']).cuda()
    betas.requires_grad = True
    # params from previous frame
    transl_prev = torch.Tensor(human_params_prev['transl']).cuda()
    global_orient_prev = torch.Tensor(human_params_prev['global_orient']).cuda()
    body_pose_prev = torch.Tensor(human_params_prev['body_pose']).cuda()

    colours = np.array(scene_mesh.visual.vertex_colors)[:, :3]
    floor = np.array([127, 127, 127], dtype='uint8')
    floor_mask = np.zeros_like(colours)
    floor_mask[:] = floor
    deform_mask = (colours != floor_mask).all(axis = 1)

    # determine the moving and static points through propagation
    distChamfer = ext.chamferDist()
    out_human = human_model.forward(transl = transl,
                                    global_orient = global_orient,
                                    body_pose = body_pose,
                                    betas = betas,
                                    **fixed_params)
    out_verts = out_human.vertices.squeeze()
    # human_con_verts = out_verts[con_all_inds]
    human_con_verts = out_verts
    scene_v_orig = torch.tensor(np.array(scene_mesh.vertices), dtype=torch.float32).unsqueeze(0).cuda()
    scene_vn_orig = torch.tensor(np.array(scene_mesh.vertex_normals), dtype=torch.float32).cuda()
    contact_body = human_con_verts.unsqueeze(0).contiguous().cuda()
    _, _, contact_scene_idx, _ = distChamfer(contact_body, scene_v_orig)
    contact_scene_idx = contact_scene_idx.squeeze().cpu().numpy()
    neighbours = find_vertex_neighbours(contact_scene_idx, scene_mesh, orders=4)
    nei_mask = np.zeros_like(deform_mask)
    nei_mask[neighbours] = 1
    nei_mask = nei_mask * deform_mask
    deform_ids = np.nonzero(nei_mask)[0].tolist()
    if np.nonzero(nei_mask)[0].shape[0] > 100:
        scene_deform_flag = 1
        static_mask = 1 - nei_mask
        static_ids = np.nonzero(static_mask)[0].tolist()
        static_pos = []
        for id in static_ids:
            static_pos.append(scene_mesh.vertices[id])
    else:
        # print('no scene deformation')
        scene_deform_flag = 0

    scene_open3d = scene_mesh.as_open3d
    scene_v = torch.tensor(np.array(scene_open3d.vertices), dtype=torch.float32).unsqueeze(0).cuda()
    scene_open3d.compute_vertex_normals()
    scene_vn = torch.tensor(np.array(scene_open3d.vertex_normals), dtype=torch.float32).cuda()

    # ================ overall ===================
    # print("optimising for all...")
    opti_params = [transl, global_orient, body_pose]
    if init:
        opti_params.append(betas)
    optimiser = torch.optim.Adam(opti_params, lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=15, gamma=0.5)
    max_iter = 30
    pbar = tqdm(total = max_iter)
    for i in range(max_iter):
        # if not scene_deform_flag:
        #     break
        out_human = human_model.forward(transl = transl,
                                        global_orient = global_orient,
                                        body_pose = body_pose,
                                        betas = betas,
                                        **fixed_params)
        out_verts = out_human.vertices.squeeze()

        # contact loss
        if raycast_results is not None:
            con_verts_human = out_verts[con_inds_human]
            loss_contact = contact_robustifier(con_verts_human - con_loc_scene).mean()
        else:
            loss_contact = 0

        # openpose reprojection loss
        if j2ds_conf_idx is not None and j2ds_conf_idx.shape[0]:
            j3d_openpose = torch.index_select(out_human.joints, 1, joint_mapper).squeeze()
            j2d = proj_op(j3d_openpose, cam_in, cam_k)
            loss_j2d = reproj_robustifier(j2d[j2ds_conf_idx] - j2ds_openpose[j2ds_conf_idx]).mean()
        else:
            loss_j2d = 0

        # anti-collision loss
        collide_ids_human, collide_ids_scene = \
        collision_check(scene_v, scene_vn, out_verts, distChamfer)

        # for collision, register to nearest surface points
        if collide_ids_scene is not None:
            loss_col = contact_robustifier(out_verts[collide_ids_human]-
                                           scene_v.squeeze()[collide_ids_scene]).mean()
        else:
            loss_col = 0

        # hidden contact loss computed by chamfer distance
        if len(con_hid_ids) == 0:
            loss_contact_hid = 0
        else:
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

        is_last = False
        if i > int(0.5 * max_iter):
            is_last = True
        if scene_deform_flag and i > 0 and i % 10 == 0:
        # if scene_deform_flag and i % 10 == 0:
            deform_ids_human, deform_ids_scene = \
            collision_check_deform(deform_mask, scene_v_orig, scene_vn_orig,
                                   out_verts, con_all_inds, human_model.faces,
                                   distChamfer, is_last = is_last)
            # for contact collision, deform the scene
            if deform_ids_scene is not None:
                handle_ids = deform_ids_scene.tolist()
                # scene_mesh.visual.vertex_colors[handle_ids] = [0, 0, 0, 0]
                handle_pos = out_verts[deform_ids_human].detach().cpu().numpy().tolist()
                constraint_ids = o3d.utility.IntVector(static_ids + handle_ids)
                constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)
                scene_deformed = scene_open3d.deform_as_rigid_as_possible(constraint_ids,
                                                                          constraint_pos,
                                                                          max_iter=10,
                                                                          energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed,
                                                                          smoothed_alpha=1e+4)
                scene_v = torch.tensor(np.array(scene_deformed.vertices), dtype=torch.float32).unsqueeze(0).cuda()
                scene_deformed.compute_vertex_normals()
                scene_vn = torch.tensor(np.array(scene_deformed.vertex_normals), dtype=torch.float32).cuda()
                # scene_open3d.vertices = scene_deformed.vertices
            else:
                # break
                a = 1

        loss = 1e+1*loss_contact + 1e-2*loss_j2d + 1e+1*loss_contact_hid
        + 1e+1*loss_tmp_trans + 1e+2*loss_tmp_ori + 1e+1*loss_tmp_pose + 1e+0*loss_col

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
    #
    # save_path = savepath + 'stage3'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # trimesh.exchange.export.export_mesh(human_mesh, os.path.join(save_path,'{:05d}'.format(frame_num)+'.ply'))
    #
    # save_path_scene = save_path + '_scene'
    # if not os.path.exists(save_path_scene):
    #     os.makedirs(save_path_scene)

    if scene_deform_flag and deform_ids_scene is not None:
        scene_mesh.vertices = np.asarray(scene_deformed.vertices)
        scene_mesh = scene_mesh.apply_transform(cam_ex)
        human_model_params['deform_ids'] = deform_ids
        human_model_params['deformed_verts'] = scene_mesh.vertices[deform_ids]

        # if raycast_results is not None:
        #     scene_mesh.visual.vertex_colors[scene_con_verts] = [255, 204, 229, 255]
        # trimesh.exchange.export.export_mesh(scene_mesh,
        #                                     os.path.join(save_path_scene,'{:05d}'.format(frame_num)+'.ply'))
    else:
        human_model_params['deform_ids'] = None
        human_model_params['deformed_verts'] = None

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

    total_frms = 0
    total_improve = 0
    for seq_name in seq_names:
        print(f'processing {seq_name}')
        data_loader = newdata_utils.load_data(seq_name)
        # avail_frms = data_loader.check_seq_avail(model='RGB')
        avail_path = './dataset/'+seq_name+'/avail_frms.npy'
        avail_frms = np.load(avail_path)
        human_model = data_loader.get_human_model()
        scene_mesh = data_loader.get_scene_mesh()
        world2cam, cam_in, cam_k = data_loader.get_cam_params()
        cam_ex = np.linalg.inv(world2cam)
        cam_R = cam_ex[:3, :3]
        cam_T = cam_ex[:3, 3]
        scene_mesh = scene_mesh.apply_transform(world2cam)
        savepath = './dataset/'+seq_name+'/'
        opt_params = np.load(savepath+'stage2.npy', allow_pickle=True).item()

        save_opt = {}
        for i, frame_num in enumerate(avail_frms):
            print(f'optimising frame {frame_num}...')
            j2ds_openpose = data_loader.load_openpose(frame_num)
            human_model_params = opt_params[str(frame_num)]
            contacts = data_loader.get_contacts(frame_num)
            raycast_results = data_loader.get_raycast_results(frame_num)

            # gt
            human_model_params_gt = data_loader.get_smplx_gt(frame_num)
            out_human_gt = human_model.forward(**human_model_params_gt)
            out_verts_gt = out_human_gt.vertices.squeeze().detach().numpy()
            # orig
            out_human = human_model.forward(**human_model_params)
            out_verts = out_human.vertices.squeeze().detach().numpy()
            out_verts = np.dot(cam_R, out_verts.T).T + cam_T

            if frame_num == avail_frms[0]:
                frame_num_prev = frame_num
                human_params_prev = optimise_pipline(frame_num, frame_num_prev, seq_name,
                                                     human_model, human_model_params,
                                                     raycast_results, cam_ex, cam_in, cam_k,
                                                     contacts, copy.deepcopy(scene_mesh),
                                                     joint_mapper, j2ds_openpose,
                                                     human_model_params, savepath, init=True)
            else:
                frame_num_prev = avail_frms[i-1]
                human_params_prev = optimise_pipline(frame_num, frame_num_prev, seq_name,
                                                     human_model, human_model_params,
                                                     raycast_results, cam_ex, cam_in, cam_k,
                                                     contacts, copy.deepcopy(scene_mesh),
                                                     joint_mapper, j2ds_openpose,
                                                     human_params_prev, savepath)

            # opt
            human_model = human_model.cpu()
            out_human_opt = human_model.forward(**human_params_prev)
            out_verts_opt = out_human_opt.vertices.squeeze().detach().numpy()
            out_verts_opt = np.dot(cam_R, out_verts_opt.T).T + cam_T

            v2v = np.mean(np.linalg.norm(out_verts_gt - out_verts, axis=1))
            v2v_opt = np.mean(np.linalg.norm(out_verts_gt - out_verts_opt, axis=1))
            total_improve = total_improve + v2v - v2v_opt
            total_frms = total_frms + 1
            print(f'v2v improve: {1000 * total_improve/total_frms}')

            save_opt[str(frame_num)] = human_params_prev
        np.save(savepath + 'stage3.npy', save_opt)
