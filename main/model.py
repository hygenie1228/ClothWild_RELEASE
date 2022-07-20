import torch
import torch.nn as nn
import numpy as np
import copy
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import ClothNet
from nets.loss import ClothClsLoss, GenderClsLoss, SdfParseLoss, SdfDPLoss, RegLoss
from utils.human_models import smpl

from config import cfg
from utils.SMPLicit import SMPLicit


class Model(nn.Module):
    def __init__(self, backbone, cloth_net, mode): 
        super(Model, self).__init__()
        self.backbone = backbone
        self.cloth_net = cloth_net
        self.mode = mode

        self.smpl_layer = [copy.deepcopy(smpl.layer['neutral']).cuda(),
                            copy.deepcopy(smpl.layer['male']).cuda(),
                            copy.deepcopy(smpl.layer['female']).cuda()]
        self.smplicit_layer = SMPLicit.SMPLicit(cfg.smplicit_path, cfg.cloth_types).cuda()

        if mode == 'train':
            self.cloth_cls_loss = ClothClsLoss()
            self.gender_cls_loss = GenderClsLoss()
            self.sdf_dp_loss = SdfDPLoss()
            self.reg_loss = RegLoss()

        self.trainable_modules = [self.backbone, self.cloth_net]

    def forward(self, inputs, targets, meta_info, mode):
        batch_size = inputs['img'].shape[0]      

        # feature extract & get cloth parameter
        img_feat = self.backbone(inputs['img'])
        pred_genders, pred_scores, z_cuts, z_styles = self.cloth_net(img_feat)
        
        # forward SMPL parameters to the SMPL layer
        smpl_pose = meta_info['smpl_pose']
        smpl_shape = meta_info['smpl_shape']
        cam_trans =  meta_info['cam_trans']

        if mode == 'train':
            # forward cloth & gender parameters to the SMPLicit layer
            smpl_gender = targets['gender']
            sdfs, cloth_meshes, cloth_meshes_unposed = self.smplicit_layer(z_cuts, z_styles, smpl_pose, smpl_shape, smpl_gender, do_marching_cube=(mode=='test'), valid=torch.ones((len(z_cuts),), dtype=torch.bool), do_smooth=False)
            
            # loss functions
            loss = {}
            loss['cloth_cls'] = cfg.cls_weight * self.cloth_cls_loss(pred_scores, targets['smpl_patch_idx'], targets['smpl_cloth_idx'])
            loss['gender_cls'] = cfg.cls_weight * self.gender_cls_loss(pred_genders, smpl_gender)
            loss['sdf_dp'] = 0.0
            loss['reg'] = 0.0
            z_cut_reg, z_style_reg = 0.0, 0.0

            for i in range(len(cfg.cloth_types)):
                cloth_type = cfg.cloth_types[i]

                if cloth_type == 'uppercloth':
                    target_cloth_idx = (i+1, cfg.cloth_types.index('coat')+1)
                else:
                    target_cloth_idx = (i+1,)

                if cloth_type == 'pants' or cloth_type == 'skirts':
                    body_pose = smpl.Astar_pose.float().cuda().repeat(batch_size,1)[:,3:]
                else:
                    body_pose = torch.zeros((batch_size,(smpl.joint_num-1)*3)).float().cuda()
                
                # DensePose based loss
                v_template = self.smpl_layer[0](global_orient=torch.zeros((batch_size,3)).float().cuda(), body_pose=body_pose, betas=smpl_shape).vertices
                loss['sdf_dp'] += cfg.dp_weight * self.sdf_dp_loss(sdfs[i], cloth_meshes_unposed[i], targets['smpl_cloth_idx'], meta_info['smpl_cloth_valid'], target_cloth_idx, cfg.sdf_thresh[cloth_type], cfg.dist_thresh[cloth_type], v_template)
                
                # Regularization loss
                cloth_exist = (sum([targets['smpl_cloth_idx'] == idx for idx in target_cloth_idx]) > 0).sum(1) > 0
                if cloth_type != 'shoes': # shoes do not have z_cut
                    z_cut_reg += cfg.cloth_reg_weight[cloth_type] * self.reg_loss(z_cuts[i], cloth_exist)
                z_style_reg += cfg.cloth_reg_weight[cloth_type] * self.reg_loss(z_styles[i], cloth_exist)
            
            loss['reg'] = cfg.reg_weight * (z_cut_reg + z_style_reg) / 2.0
            
            return loss
        else:
            pred_clothes = []
            pred_gender = []
            cloth_meshes = []
            
            for i in range(batch_size):
                z_cut = []; z_style = []
                for j in range(len(cfg.cloth_types)):                        
                    z_cut.append(z_cuts[j][i][None,:])
                    z_style.append(z_styles[j][i][None,:])
                
                valid_clothes = pred_scores[i] > cfg.cls_threshold
                gender = torch.argmax(pred_genders[i])+1  # male:1, female:2
                
                _, cloth_mesh, _ = self.smplicit_layer(z_cut, z_style, smpl_pose[None, i], smpl_shape[None, i], [gender], True, valid=valid_clothes)
                
                pred_clothes.append(valid_clothes)
                cloth_meshes.append(cloth_mesh)
                pred_gender.append(gender)

            cloth_meshes = [[i[0] for i in clothmesh] for clothmesh in zip(*cloth_meshes)]
            
            # add camera translations
            for i in range(len(cfg.cloth_types)):
                for j in range(batch_size):
                    if cloth_meshes[i][j] is not None:
                        cloth_meshes[i][j].vertices += cam_trans[j].detach().cpu().numpy()
            
            mesh_cam = self.get_coords(smpl_pose[:,:3], {'shape': smpl_shape, 'pose': smpl_pose[:,3:]}, cam_trans, pred_gender)
            
            # output
            out = {} 
            out['pred_clothes'] = pred_clothes
            out['pred_gender'] = pred_gender
            out['smpl_mesh'] = mesh_cam

            for i,cloth_type in enumerate(cfg.cloth_types):
                out[cloth_type + '_mesh'] = cloth_meshes[i]
            
            for k,v in targets.items():
                out[f'{k}_target'] = v
            
            return out

    def get_coords(self, root_pose, params, cam_trans, gender):
        batch_size = root_pose.shape[0]

        if self.mode != 'train':
            mesh_cam = []
            for i in range(batch_size):
                output = self.smpl_layer[gender[i]](betas=params['shape'][None,i], body_pose=params['pose'][None,i], global_orient=root_pose[None,i], transl=cam_trans[None,i])
                mesh_cam.append(output.vertices)
            mesh_cam = torch.cat(mesh_cam, dim=0)
        else:
            output = self.smpl_layer[0](betas=params['shape'], body_pose=params['pose'], global_orient=root_pose, transl=cam_trans)
            mesh_cam = output.vertices

        return mesh_cam


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)


def get_model(mode):
    backbone = ResNetBackbone(cfg.resnet_type)
    cloth_net = ClothNet()

    if mode == 'train':
        backbone.init_weights()
        cloth_net.apply(init_weights)

    model = Model(backbone, cloth_net, mode)
    return model
