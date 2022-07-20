import torch
import json
import sys
import numpy as np
from .util_smpl import batch_global_rigid_transformation, batch_rodrigues, reflect_pose
import torch.nn as nn
import os
import trimesh
import pickle
#from utils.human_models import smpl
#from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_parse, vis_dp

class SMPL(nn.Module):
    def __init__(self, model_path, joint_type = 'cocoplus', obj_saveable = False):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)

        self.model_path = model_path
        self.joint_type = joint_type
        
        with open(model_path, 'rb') as reader:
            model = pickle.load(reader, encoding='latin1')
            
        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None

        np_v_template = np.array(model['v_template'], dtype = np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype = np.float)[:,:,:10]
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = np.array(model['J_regressor'].toarray().transpose(1,0), dtype = np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = np.array(model['posedirs'], dtype = np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_weights = np.array(model['weights'], dtype = np.float)

        vertex_count = np_weights.shape[0] 
        vertex_component = np_weights.shape[1]

        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))
        self.register_buffer('e3', torch.eye(3).float())
        
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write( 'v {:f} {:f} {:f}\n'.format( v[0], v[1], v[2]) )

            for f in self.faces: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f {:d} {:d} {:d}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1) )

    def forward(self, beta, theta, get_skin = False, theta_in_rodrigues=True):
        device, dtype = beta.device, beta.dtype
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :] - torch.eye(3, dtype=dtype, device=device)).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W=self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        
        #v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def deform_clothed_smpl(self, theta, J, v_smpl, v_cloth):
        num_batch = theta.shape[0]

        device = theta.device
        self.cur_device = torch.device(device.type, device.index)
        
        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :] - torch.eye(3, device=device).float()).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_smpl = pose_params + v_smpl
        
        # Calculate closest SMPL vertex for each vertex of the cloth mesh
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth.unsqueeze(2))**2).sum(-1)
            dists, correspondance = torch.min(dists, 2) # num_batch, v_cloth.shape[1]

        v_posed_cloth = torch.gather(pose_params, 1, correspondance[:,:,None].repeat(1,1,3)) + v_cloth
        J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W = self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_smpl = torch.cat([v_posed_smpl, torch.ones(num_batch, v_posed_smpl.shape[1], 1, device=device)], dim = 2)
        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device=device)], dim = 2)
        v_homo_smpl = torch.matmul(T, torch.unsqueeze(v_posed_homo_smpl, -1))
        v_homo_cloth = torch.matmul(torch.gather(T, 1, correspondance[:,:,None,None].repeat(1,1,4,4)), torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_smpl = v_homo_smpl[:, :, :3, 0]
        verts_cloth = v_homo_cloth[:, :, :3, 0]
        return verts_cloth
   
    def unpose_and_deform_cloth(self, v_cloth_posed, theta_from, theta_to, beta, Jsmpl, vsmpl, theta_in_rodrigues=True):
        ### UNPOSE:
        device = theta_from.device
        self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if theta_in_rodrigues:
            Rs = batch_rodrigues(theta_from.view(-1, 3)).view(-1, 24, 3, 3)
        else: #theta is already rotations
            Rs = theta_from.view(-1,24,3,3)

        pose_feature = (Rs[:, 1:, :, :] - torch.eye(3, device=device).float()).view(-1, 207)

        pose_displ = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed = pose_displ + v_shaped
        J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        W = self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        v_smpl = v_homo[:, :, :3, 0]
        with torch.no_grad():
            dists = ((v_smpl.unsqueeze(1) - v_cloth_posed.unsqueeze(2))**2).sum(-1)
            dists, correspondance = torch.min(dists, 2) # num_batch, v_cloth_posed.shape[1]

        invT = torch.inverse(torch.gather(T, 1, correspondance[:,:,None,None].repeat(1,1,4,4)).view(num_batch,-1,4,4))
        v = torch.cat([v_cloth_posed, torch.ones(num_batch, v_cloth_posed.shape[1], 1, device=device)], 2)
        v = torch.matmul(invT, v.unsqueeze(-1))[:,:, :3, 0]
        unposed_v = v - torch.gather(pose_displ, 1, correspondance[:,:,None].repeat(1,1,3))
        
        ### REPOSE:
        Rs = batch_rodrigues(theta_to.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :] - torch.eye(3, device=device).float()).view(-1, 207)

        pose_params = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
        v_posed_cloth = torch.gather(pose_params,1,correspondance[:,:,None].repeat(1,1,3)) + unposed_v
        J_transformed, A = batch_global_rigid_transformation(Rs, Jsmpl, self.parents, rotate_base = False)

        W = self.weight.expand(num_batch,*self.weight.shape[1:])
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo_cloth = torch.cat([v_posed_cloth, torch.ones(num_batch, v_posed_cloth.shape[1], 1, device=device)], dim = 2)
        v_homo_cloth = torch.matmul(torch.gather(T,1,correspondance[:,:,None,None].repeat(1,1,4,4)), torch.unsqueeze(v_posed_homo_cloth, -1))
        verts_cloth = v_homo_cloth[:, :, :3, 0]
        return verts_cloth


    def skeleton(self,beta,require_body=False):
        num_batch = beta.shape[0]
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        if require_body:
            return J, v_shaped
        else:
            return J
