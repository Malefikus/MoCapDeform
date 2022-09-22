import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh

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

def find_vertex_neighbours(query_idx, scene_mesh, orders=1):
    for order in range(orders):
        neighbours = []
        for idx in range(query_idx.shape[0]):
            query = query_idx[idx]
            neis = scene_mesh.vertex_neighbors[query]
            neighbours.append(query)
            neighbours.extend(neis)
        neighbours = np.array(list(set(neighbours)))
        query_idx = neighbours
    return neighbours

def collision_check(scene_v, scene_vn, human_vertices, distChamfer):
    # compute the full-body chamfer distance
    human_vertices = human_vertices.unsqueeze(0).contiguous()
    contact_dist, _, contact_scene_idx, _ = distChamfer(human_vertices, scene_v)
    contact_scene_idx = contact_scene_idx.squeeze().cpu().numpy()
    # generate collision mask via normal check
    human2scene_norm = scene_v.squeeze()[contact_scene_idx] - human_vertices.squeeze()
    # human2scene_norm = human2scene_norm/torch.norm(human2scene_norm, dim=1, keepdim=True)
    human2scene_norm = F.normalize(human2scene_norm, p=2, dim=1)
    scene_norm = scene_vn[contact_scene_idx]
    collide_mask = torch.sum(human2scene_norm * scene_norm, dim=1)
    collide_mask = collide_mask > 0

    collide_ids_human = torch.nonzero(collide_mask.squeeze())
    # return collide_ids_human and collide_ids_scene
    if collide_ids_human.numel():
        collide_ids_human = collide_ids_human.squeeze().cpu().numpy()
        collide_ids_scene = contact_scene_idx[collide_ids_human]
    else:
        collide_ids_human = None
        collide_ids_scene = None

    return collide_ids_human, collide_ids_scene

def collision_check_deform(deform_mask, scene_v, scene_vn,
                           human_vertices, con_all_inds, human_faces,
                           distChamfer, is_last = False):
    # for deformable use, collision should be checked with original vertices
    # compute the full-body chamfer distance
    human_vertices = human_vertices.unsqueeze(0).contiguous()
    contact_dist, _, contact_scene_idx, _ = distChamfer(human_vertices, scene_v)
    contact_scene_idx = contact_scene_idx.squeeze().cpu().numpy()
    # generate collision mask via normal check
    human2scene_norm = scene_v.squeeze()[contact_scene_idx] - human_vertices.squeeze()
    # human2scene_norm = human2scene_norm/torch.norm(human2scene_norm, dim=1, keepdim=True)
    human2scene_norm = F.normalize(human2scene_norm, p=2, dim=1)
    scene_norm = scene_vn[contact_scene_idx]
    collide_mask = torch.sum(human2scene_norm * scene_norm, dim=1)
    collide_mask = collide_mask > 0

    # generate scene-human normal mask (to prevent spikes)
    out_verts = human_vertices.squeeze().detach().cpu().numpy()
    human_mesh = trimesh.base.Trimesh(vertices=out_verts, faces=human_faces, process=False)
    human_vn = torch.tensor(np.array(human_mesh.vertex_normals), dtype=torch.float32).cuda()
    human_norm_mask = torch.sum(human_vn * scene_norm, dim=1)
    human_norm_mask = human_norm_mask < -0.8

    collide_mask = collide_mask * human_norm_mask
    collide_ids_human = torch.nonzero(collide_mask.squeeze())

    if not collide_ids_human.numel():
        return None, None

    collide_ids_human = collide_ids_human.squeeze().cpu().numpy()
    # filter out the non-deformable points on scene
    collide_scene_mask = np.zeros_like(deform_mask)
    collide_scene_mask[contact_scene_idx[collide_ids_human]] = 1
    collide_scene_mask = collide_scene_mask * deform_mask
    if not is_last:
        contact_scene_mask = np.zeros_like(deform_mask)
        contact_scene_mask[contact_scene_idx[con_all_inds]] = 1
        collide_scene_mask = collide_scene_mask * contact_scene_mask

    collide_human_mask = collide_scene_mask[contact_scene_idx[collide_ids_human]]

    collide_human_mask = torch.tensor(collide_human_mask)
    if torch.nonzero(collide_human_mask).numel() < 2:
        return None, None

    col_con_human = collide_ids_human[torch.nonzero(collide_human_mask)].reshape(-1)

    # # for vis purpose only
    human_mesh = trimesh.base.Trimesh(vertices=out_verts, faces=human_faces, process=False)
    col_con_human = find_vertex_neighbours(col_con_human, human_mesh, orders=2)

    col_con_scene = contact_scene_idx[col_con_human].reshape(-1)

    return col_con_human, col_con_scene
