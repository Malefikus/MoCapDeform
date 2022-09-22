import os
import numpy as np
from smplx import SMPLX
import pdb


class SMPLXProcessor():
    def __init__(self, human_model_path):

        self.human_model_path = human_model_path

        self.smplx_body_pose_parameter_order = {"left_hip": 0, "right_hip": 1,
                            "low_spine": 2,
                            "left_knee": 3, "right_knee": 4,
                            "middle_spine": 5,
                            "left_ankle": 6, "right_ankle": 7,
                            "high_spine": 8,
                            "left_toe": 9, "right_toe": 10,
                            "low_neck": 11,
                            "left_clavicle": 12, "right_clavicle": 13,
                            "high_neck": 14,
                            "left_shoulder": 15, "right_shoulder": 16,
                            "left_elbow": 17, "right_elbow": 18,
                            "left_wrist": 19, "right_wrist": 20}

        self.smplx_hand_pose_parameter_order = {
                            "index1": 0, "index2": 1, "index3": 2,
                            "middle1": 3, "middle2": 4, "middle3": 5,
                            "pinky1": 6, "pinky2": 7, "pinky3": 8,
                            "ring1": 9, "ring2": 10, "ring3": 11,
                            "thumb1": 12, "thumb2": 13, "thumb3": 14}

        self.smplx_joint_order = {
                            "base": 0,
                            "left_hip": 1, "right_hip": 2,
                            "low_spine": 3,
                            "left_knee": 4, "right_knee": 5,
                            "middle_spine": 6,
                            "left_ankle": 7, "right_ankle": 8,
                            "high_spine": 9,

                            "left_toe": 10, "right_toe": 11,
                            "low_neck": 12,
                            "left_clavicle": 13, "right_clavicle": 14,
                            "high_neck": 15,
                            "left_shoulder": 16, "right_shoulder": 17,

                            "left_elbow": 18,"right_elbow": 19,
                            "left_wrist": 20, "right_wrist": 21,

                            # not sure
                            "head_top": 24,

                            "left_thumb1": 37, "left_thumb2": 38, "left_thumb3": 39,
                            "left_index1": 25, "left_index2": 26, "left_index3": 27,
                            "left_middle1": 28, "left_middle2": 29, "left_middle3": 30,
                            "left_ring1": 34, "left_ring2": 35, "left_ring3": 36,
                            "left_pinky1": 31, "left_pinky2": 32, "left_pinky3": 33,

                            "right_thumb1": 52, "right_thumb2": 53, "right_thumb3": 54,
                            "right_index1": 40, "right_index2": 41, "right_index3": 42,
                            "right_middle1": 43, "right_middle2": 44, "right_middle3": 45,
                            "right_ring1": 49, "right_ring2": 50, "right_ring3": 51,
                            "right_pinky1": 46, "right_pinky2": 47, "right_pinky3": 48
                            }

        self.smplx_body_dof = {
                            "left_hip":[1, 1, 1], "right_hip":[1, 1, 1],
                            "low_spine":[0, 0, 0],
                            "left_knee":[1, 0, 0], "right_knee":[1, 0, 0],
                            "middle_spine":[0, 0, 0],
                            # "middle_spine":[1, 1, 1],
                            "left_ankle":[1, 1, 0], "right_ankle":[1, 1, 0],
                            "high_spine":[0, 0, 0],
                            "left_toe":[0, 0, 0], "right_toe": [0, 0, 0],
                            "low_neck":[1, 1, 1],
                            # "left_clavicle":[0, 1, 1], "right_clavicle":[0, 1, 1],
                            "left_clavicle":[1, 1, 1], "right_clavicle":[1, 1, 1],
                            "high_neck":[1, 1, 1],
                            # "left_shoulder":[0, 1, 1], "right_shoulder":[0, 1, 1],
                            "left_shoulder":[1, 1, 1], "right_shoulder":[1, 1, 1],
                            "left_elbow":[1, 1, 0], "right_elbow":[1, 1, 0],
                            "left_wrist":[0, 1, 1], "right_wrist":[0, 1, 1]
                            }

        self.smplx_body_limits_l = {
                            "left_hip":[-2.5, -0.6, -0.8],
                            "right_hip":[-2.5, -0.6, -1.4],
                            "low_spine":[0, 0, 0],
                            "left_knee":[-0.1, 0, 0],
                            "right_knee":[-0.1, 0, 0],
                            "middle_spine":[0, 0, 0],
                            "left_ankle":[-0.4, -0.5, 0],
                            "right_ankle":[-0.4, -0.5, 0],
                            "high_spine":[0, 0, 0],
                            "left_toe":[0, 0, 0],
                            "right_toe": [0, 0, 0],
                            "low_neck":[-0.6, -0.6, -0.3],
                            # "left_clavicle":[0, -0.6, -0.4],
                            # "right_clavicle":[0, -0.4, -0.4],
                            "left_clavicle":[0, 0, 0],
                            "right_clavicle":[0, 0, 0],
                            "high_neck":[-0.6, -0.6, -0.3],
                            # "left_shoulder":[0, -2, -1],
                            "left_shoulder":[0, -1.5, -1],
                            "right_shoulder":[0, -0.6, -1],
                            "left_elbow":[-1.4, -2.7, 0],
                            "right_elbow":[-1.4, 0, 0],
                            "left_wrist":[0, -0.4, -0.9],
                            "right_wrist":[0, -0.4, -0.9],
                            }

        self.smplx_body_limits_u = {
                            "left_hip":[1.4, 0.6, 1.4],
                            "right_hip":[1.4, 0.6, 0.8],
                            "low_spine":[0, 0, 0],
                            "left_knee":[2.5, 0, 0],
                            "right_knee":[2.5, 0, 0],
                            "middle_spine":[0, 0, 0],
                            "left_ankle":[0.7, 0.5, 0],
                            "right_ankle":[0.7, 0.5, 0],
                            "high_spine":[0, 0, 0],
                            "left_toe":[0, 0, 0],
                            "right_toe": [0, 0, 0],
                            "low_neck":[0.6, 0.6, 0.3],
                            # "left_clavicle":[0, 0.4, 0.4],
                            # "right_clavicle":[0, 0.6, 0.4],
                            "left_clavicle":[0, 0, 0],
                            "right_clavicle":[0, 0, 0],
                            "high_neck":[0.6, 0.6, 0.3],
                            "left_shoulder":[0, 0.6, 1],
                            # "right_shoulder":[0, 2, 1],
                            "right_shoulder":[0, 1.5, 1],
                            "left_elbow":[0.9, 0, 0],
                            "right_elbow":[0.9, 2.7, 0],
                            "left_wrist":[0, 0.4, 0.9],
                            "right_wrist":[0, 0.4, 0.9],
                            }

        self.smplx_hand_dof = {
                            "index1": [0, 1, 1], "index2": [0, 0, 1], "index3": [0, 0, 1],
                            "middle1": [0, 1, 1], "middle2": [0, 0, 1], "middle3": [0, 0, 1],
                            "pinky1": [0, 1, 1], "pinky2": [0, 0, 1], "pinky3": [0, 0, 1],
                            "ring1": [0, 1, 1], "ring2": [0, 0, 1], "ring3": [0, 0, 1],
                            "thumb1": [1, 1, 1], "thumb2": [0, 0, 1], "thumb3": [0, 1, 0]
                            }

        self.smplx_lhand_limits_l = {
                            "left_index1": [0, -0.3, -0.7], "left_index2": [0, 0, -0.7], "left_index3": [0, 0, -0.7],
                            "left_middle1": [0, -0.3, -0.7], "left_middle2": [0, 0, -0.7], "left_middle3": [0, 0, -0.7],
                            "left_pinky1": [0, -0.3, -0.7], "left_pinky2": [0, 0, -0.7], "left_pinky3": [0, 0, -0.7],
                            "left_ring1": [0, -0.3, -0.7], "left_ring2": [0, 0, -0.7], "left_ring3": [0, 0, -0.7],
                            "left_thumb1": [-1, -0.7, -0.9], "left_thumb2": [0, 0, -0.7], "left_thumb3": [0, -0.3, 0]
                            }

        self.smplx_lhand_limits_u = {
                            "left_index1": [0, 0.3, 0.7], "left_index2": [0, 0, 0.7], "left_index3": [0, 0, 0.3],
                            "left_middle1": [0, 0.3, 0.7], "left_middle2": [0, 0, 0.7], "left_middle3": [0, 0, 0.3],
                            "left_pinky1": [0, 0.3, 0.7], "left_pinky2": [0, 0, 0.7], "left_pinky3": [0, 0, 0.3],
                            "left_ring1": [0, 0.3, 0.7], "left_ring2": [0, 0, 0.7], "left_ring3": [0, 0, 0.3],
                            "left_thumb1": [0.7, 0.4, 0.9], "left_thumb2": [0, 0, -0.7], "left_thumb3": [0, 0.7, 0]
                            }

        self.smplx_rhand_limits_l = {
                            "right_index1": [0, -0.3, -0.7], "right_index2": [0, 0, -0.7], "right_index3": [0, 0, -0.3],
                            "right_middle1": [0, -0.3, -0.7], "right_middle2": [0, 0, -0.7], "right_middle3": [0, 0, -0.3],
                            "right_pinky1": [0, -0.3, -0.7], "right_pinky2": [0, 0, -0.7], "right_pinky3": [0, 0, -0.3],
                            "right_ring1": [0, -0.3, -0.7], "right_ring2": [0, 0, -0.7], "right_ring3": [0, 0, -0.3],
                            "right_thumb1": [-0.7, -0.4, -0.9], "right_thumb2": [0, 0, -0.7], "right_thumb3": [0, -0.7, 0]
                            }

        self.smplx_rhand_limits_u = {
                            "right_index1": [0, 0.3, 0.7], "right_index2": [0, 0, 0.7], "right_index3": [0, 0, 0.7],
                            "right_middle1": [0, 0.3, 0.7], "right_middle2": [0, 0, 0.7], "right_middle3": [0, 0, 0.7],
                            "right_pinky1": [0, 0.3, 0.7], "right_pinky2": [0, 0, 0.7], "right_pinky3": [0, 0, 0.7],
                            "right_ring1": [0, 0.3, 0.7], "right_ring2": [0, 0, 0.7], "right_ring3": [0, 0, 0.7],
                            "right_thumb1": [1, 0.7, 0.9], "right_thumb2": [0, 0, -0.7], "right_thumb3": [0, 0.3, 0]
                            }

        self.torso_joints = ["base", "left_hip", "right_hip", "low_neck",
                             "left_clavicle", "right_clavicle",
                             "left_shoulder", "right_shoulder"]

    def get_joint_sets(self, target_joints):
        smplx_joint_order = self.smplx_joint_order

        smplx_jids = [smplx_joint_order[key] for key in target_joints]

        smplx_body_dof = self.smplx_body_dof
        smplx_body_dofs = [smplx_body_dof[key] for key in smplx_body_dof.keys()]
        smplx_body_dofs = np.array(smplx_body_dofs).reshape(-1)
        smplx_body_dofs = np.nonzero(smplx_body_dofs)[0].tolist()

        smplx_hand_dof = self.smplx_hand_dof
        smplx_hand_dofs = [smplx_hand_dof[key] for key in smplx_hand_dof.keys()]
        smplx_hand_dofs = np.array(smplx_hand_dofs).reshape(-1)
        smplx_hand_dofs = np.nonzero(smplx_hand_dofs)[0].tolist()

        torso_jids = [smplx_joint_order[key] for key in self.torso_joints]

        smplx_body_limits_l = self.smplx_body_limits_l
        smplx_body_limits_u = self.smplx_body_limits_u
        smplx_body_limits_l = [smplx_body_limits_l[key] for key in smplx_body_limits_l.keys()]
        smplx_body_limits_l = np.array(smplx_body_limits_l).reshape(-1)
        smplx_body_limits_u = [smplx_body_limits_u[key] for key in smplx_body_limits_u.keys()]
        smplx_body_limits_u = np.array(smplx_body_limits_u).reshape(-1)

        smplx_lhand_limits_l = self.smplx_lhand_limits_l
        smplx_lhand_limits_u = self.smplx_lhand_limits_u
        smplx_lhand_limits_l = [smplx_lhand_limits_l[key] for key in smplx_lhand_limits_l.keys()]
        smplx_lhand_limits_l = np.array(smplx_lhand_limits_l).reshape(-1)
        smplx_lhand_limits_u = [smplx_lhand_limits_u[key] for key in smplx_lhand_limits_u.keys()]
        smplx_lhand_limits_u = np.array(smplx_lhand_limits_u).reshape(-1)

        smplx_rhand_limits_l = self.smplx_rhand_limits_l
        smplx_rhand_limits_u = self.smplx_rhand_limits_u
        smplx_rhand_limits_l = [smplx_rhand_limits_l[key] for key in smplx_rhand_limits_l.keys()]
        smplx_rhand_limits_l = np.array(smplx_rhand_limits_l).reshape(-1)
        smplx_rhand_limits_u = [smplx_rhand_limits_u[key] for key in smplx_rhand_limits_u.keys()]
        smplx_rhand_limits_u = np.array(smplx_rhand_limits_u).reshape(-1)

        return smplx_jids, smplx_body_dofs, smplx_hand_dofs, torso_jids,\
    smplx_body_limits_l, smplx_body_limits_u, smplx_lhand_limits_l,\
    smplx_lhand_limits_u, smplx_rhand_limits_l, smplx_rhand_limits_u

    def get_smplx_instance(self):
        return SMPLX(self.human_model_path, num_pca_comps = 12, create_global_orient = True)

    def ply_write(self, vertices, faces, save_dir, filename, con_states = None):
        vertex = len(vertices)
        statements = [
            "ply",
            'format ascii 1.0',
            'element vertex ' + str(vertex),
            'property double x',
            'property double y',
            'property double z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'element face '+ str(len(faces)),
            'property list uchar int vertex_indices',
            'end_header'
        ]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        statements = statements
        f = open(save_dir + filename, 'w')
        for line in statements:
            f.write("%s\n" % line)
        for vert, line in enumerate(vertices):
            if con_states is None or con_states[vert] == 0:
                f.write("%s\n" % (" ".join(map(str, (line))) + ' ' + "128 128 128"))
            else:
                f.write("%s\n" % (" ".join(map(str, (line))) + ' ' + "0 0 256"))
        for line in faces:
            f.write("%s\n" % ("3 "+" ".join(map(str, (line)))))
        f.close()
        return 0

    def jointmap_openpose(self, use_hands=False, use_face=False):
        body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                 63, 64, 65], dtype=np.int32)
        mapping = [body_mapping]
        if use_hands:
            lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                      67, 28, 29, 30, 68, 34, 35, 36, 69,
                                      31, 32, 33, 70], dtype=np.int32)
            rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                      43, 44, 45, 73, 49, 50, 51, 74, 46,
                                      47, 48, 75], dtype=np.int32)

            mapping += [lhand_mapping, rhand_mapping]
        if use_face:
            face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                     dtype=np.int32)
            mapping += [face_mapping]

        return np.concatenate(mapping)
