import numpy as np
import torch
import os.path as osp
from config import cfg
from utils.transforms import  transform_joint_to_other_db
import smplx

class SMPL(object):
    def __init__(self):
        self.layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(cfg.human_model_path, 'smpl', gender='NEUTRAL', **self.layer_arg), 'male': smplx.create(cfg.human_model_path, 'smpl', gender='MALE', **self.layer_arg), 'female': smplx.create(cfg.human_model_path, 'smpl', gender='FEMALE', **self.layer_arg)}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10

        # SMPL joint set
        self.joint_num = 24
        self.joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
        self.flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) )
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.joint_regressor = self.layer['neutral'].J_regressor.numpy().astype(np.float32)

        # Astar pose
        self.Astar_pose = torch.zeros(1, self.joint_num*3)
        self.Astar_pose[0, 5] = 0.04 
        self.Astar_pose[0, 8] = -0.04

    def get_custom_template_layer(self, v_template, gender):
        layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False, 'v_template': v_template}
        layer = smplx.create(cfg.human_model_path, 'smpl', gender=gender.upper(), **layer_arg)
        return layer

smpl = SMPL()
