import cv2
import matplotlib.image as imgplt
import numpy as np
import trimesh
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
import copy
import newdata_utils
import pdb

class Raycast_Operation():
    def __init__(self, human_model, human_model_params, contacts,
                  scene, world2cam, cam_in, cam_k, img_path, #iuv_decomp,
                  frame_num, seq_name, mask):
        # get the image
        self.img = imgplt.imread(img_path)
        self.img_H, self.img_W, _ = self.img.shape
        self.human_model = human_model
        self.human_model_params = human_model_params
        self.contacts = contacts
        self.scene = scene
        self.cam2world = np.linalg.inv(world2cam)
        self.cam_in = cam_in
        self.cam_k = cam_k
        self.img_path = img_path
        self.frame_num = frame_num
        self.seq_name = seq_name
        self.mask = np.zeros((self.img_H, self.img_W))
        if mask is not None:
            self.mask[mask[:,0], mask[:,1]] = 1

    def rcnn_mask(self, con_points, con_ids):
        con_map = np.zeros((self.img_H, self.img_W))
        # con_points are W * H
        con_map[con_points[:,1], con_points[:, 0]] = con_ids
        masked_map = np.multiply(con_map, self.mask)
        masked_con = np.nonzero(masked_map)
        con_points_masked = np.concatenate((masked_con[1].reshape(-1, 1),
                                            masked_con[0].reshape(-1, 1)), axis = 1)
        con_ids_masked = masked_map[masked_con[0], masked_con[1]].astype(int)
        if con_points_masked.shape[0] == 0:
            con_points_masked, con_ids_masked = None, None
        return con_points_masked, con_ids_masked

    def canonicalize_2Ds(self, Ki, Kp, p_2Ds):
        # using undistortion function
        uv_orig = np.expand_dims(p_2Ds, axis=1).astype('float32')
        canonical_2Ds = cv2.undistortPoints(uv_orig, Ki, Kp).reshape(-1, 2)

        can_p_3D = np.concatenate((canonical_2Ds, np.ones((len(canonical_2Ds), 1))), 1)

        return can_p_3D

    def pick_pnts_3d(self, points, con_is):
        clustering  = DBSCAN(eps = 0.3, min_samples = 2).fit(points)
        cluster_labels = clustering.labels_
        con_i_types = list(set(con_is.tolist()))
        flag = 0
        for i_num in con_i_types:
            sub_inds = np.argwhere(con_is == i_num).reshape(-1)
            sub_clusters = cluster_labels[sub_inds]
            sub_points = points[sub_inds]
            pnts_norm = np.linalg.norm(sub_points, axis = 1)
            min_pnt_label = sub_clusters[pnts_norm.argmin()]
            sub_inds_picked = np.argwhere(sub_clusters == min_pnt_label).reshape(-1)
            if flag == 0:
                inds_picked = sub_inds[sub_inds_picked]
                flag = 1
            else:
                inds_picked = np.concatenate((inds_picked, sub_inds[sub_inds_picked]))
        return inds_picked

    def pick_pnts_3d_all(self, points):
        # perform dbscan to delete the far points
        clustering  = DBSCAN(eps = 0.5, min_samples = 2).fit(points)
        cluster_labels = clustering.labels_

        pnts_norm = np.linalg.norm(points, axis=1)
        min_pnt_label = cluster_labels[pnts_norm.argmin()]
        inds_picked = np.argwhere(cluster_labels == min_pnt_label).reshape(-1)

        if inds_picked.shape[0] < 50:
            return None
        else:
            # print(inds_picked.shape[0])
            return inds_picked

    def find_intersects(self, con_points, con_ids_human, con_is, human_mesh):

        vectors_con = self.canonicalize_2Ds(self.cam_in, self.cam_k, con_points)
        origins_con = np.zeros_like(vectors_con)

        # do the actual ray- mesh queries
        # points, index_ray, index_tri = scene_mesh.ray.intersects_location(origins_con, vectors_con, multiple_hits = True)
        # install pyembree to speed up by 50x
        points, index_ray, index_tri = scene_mesh.ray.intersects_location(origins_con, vectors_con, multiple_hits = False)

        if points.shape[0] == 0:
            return None

        inds_picked = self.pick_pnts_3d_all(points)
        if inds_picked is None:
            return inds_picked

        index_tri_picked = index_tri[inds_picked]
        con_pnts_loc_scene = points[inds_picked]
        index_ray_picked = index_ray[inds_picked]
        con_pnts_img = con_points[index_ray_picked]
        con_ids_human = con_ids_human[index_ray_picked]

        # save the raycast results
        results = {'face_inds_scene': index_tri_picked,
                   'pnts_loc_scene': con_pnts_loc_scene,
                   'pix_loc_img': con_pnts_img,
                   'vert_inds_human': con_ids_human}

        print(f'processing frame {self.frame_num}')

        return results

    def execute(self):
        # human models. camera coordinates
        human_model_posed = self.human_model.forward(return_verts=True, **self.human_model_params)
        human_verts = np.array(human_model_posed.vertices.view(-1, 3).detach())
        human_mesh = trimesh.base.Trimesh(vertices=human_verts, faces=self.human_model.faces, process=False)

        proj_points = cv2.projectPoints(human_verts, np.eye(3), np.zeros(3), self.cam_in, self.cam_k)[0].squeeze()
        proj_points[:, 0] = np.clip(proj_points[:, 0], 0, self.img_W - 1)
        proj_points[:, 1] = np.clip(proj_points[:, 1], 0, self.img_H - 1)

        contacts = self.contacts > 0.5
        con_ids = np.nonzero(contacts)[0]

        con_points = proj_points[con_ids].astype(int)

        con_points_masked, con_ids_masked = self.rcnn_mask(con_points, con_ids)
        con_is = None

        if con_points_masked is None:
            results = None
        else:
            results = self.find_intersects(con_points_masked, con_ids_masked, con_is, human_mesh)

        return results

if __name__ == '__main__':
    seq_names = ['s1', 's2', 's3', 's4']
    for seq_name in seq_names:
        print(f'processing {seq_name}')
        data_loader = newdata_utils.load_data(seq_name)
        avail_path = './dataset/'+seq_name+'/avail_frms.npy'
        avail_frms = np.load(avail_path)
        human_model = data_loader.get_human_model()
        world2cam, cam_in, cam_k = data_loader.get_cam_params()
        scene_mesh = data_loader.get_scene_mesh()
        scene_mesh = scene_mesh.apply_transform(world2cam)
        results_seq = {}
        for frame_num in avail_frms:
            human_model_params = data_loader.get_human_model_params(frame_num, 'RGB')
            contacts = data_loader.get_contacts(frame_num)
            img_path = data_loader.get_image_path(frame_num)
            mask = data_loader.get_mask(frame_num)

            # visualise the frame
            rayop = Raycast_Operation(copy.deepcopy(human_model), human_model_params,
                                            contacts, copy.deepcopy(scene_mesh), world2cam,
                                            cam_in, cam_k, img_path,
                                            frame_num, seq_name, mask)
            results = rayop.execute()
            results_seq[str(frame_num)] = results

        save_path = './dataset/' + seq_name + '/raycast.npy'
        np.save(save_path, results_seq)
