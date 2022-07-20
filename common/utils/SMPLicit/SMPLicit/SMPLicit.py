import torch
import numpy as np
import torch.nn as nn
import os
import os.path as osp
import trimesh
import math
import copy
from .SMPL import SMPL
from .SMPLicit_options import Options
from .smplicit_core_test import Model

class SMPLicit(nn.Module):
    def __init__(self, root_path, cloth_types):
        super(SMPLicit, self).__init__()
        self._opt = Options()

        uppercloth = Model(osp.join(root_path, self._opt.path_checkpoints, 'upperclothes.pth'),
                                        self._opt.upperbody_n_z_cut, 
                                        self._opt.upperbody_n_z_style, self._opt.upperbody_num_clusters, 
                                        osp.join(root_path, self._opt.path_cluster_files, self._opt.upperbody_clusters), 
                                        self._opt.upperbody_b_min, self._opt.upperbody_b_max,
                                        self._opt.upperbody_resolution, thresh=self._opt.upperbody_thresh_occupancy)

        coat = Model(osp.join(root_path, self._opt.path_checkpoints, 'upperclothes.pth'),
                                        self._opt.upperbody_n_z_cut, 
                                        self._opt.upperbody_n_z_style, self._opt.upperbody_num_clusters, 
                                        osp.join(root_path, self._opt.path_cluster_files, self._opt.upperbody_clusters), 
                                        self._opt.upperbody_b_min, self._opt.upperbody_b_max,
                                        self._opt.upperbody_resolution, thresh=self._opt.coat_thresh_occupancy)


        pants = Model(osp.join(root_path, self._opt.path_checkpoints, 'pants.pth'),
                                        self._opt.pants_n_z_cut, 
                                        self._opt.pants_n_z_style, self._opt.pants_num_clusters, 
                                        osp.join(root_path, self._opt.path_cluster_files, self._opt.pants_clusters), 
                                        self._opt.pants_b_min, self._opt.pants_b_max,
                                        self._opt.pants_resolution, thresh=self._opt.pants_thresh_occupancy)

        skirts = Model(osp.join(root_path, self._opt.path_checkpoints, 'skirts.pth'),
                                        self._opt.skirts_n_z_cut, 
                                        self._opt.skirts_n_z_style, self._opt.skirts_num_clusters, 
                                        osp.join(root_path, self._opt.path_cluster_files, self._opt.skirts_clusters), 
                                        self._opt.skirts_b_min, self._opt.skirts_b_max,
                                        self._opt.skirts_resolution, thresh=self._opt.skirts_thresh_occupancy)

        hair = Model(osp.join(root_path, self._opt.path_checkpoints, 'hair.pth'),
                                        self._opt.hair_n_z_cut, 
                                        self._opt.hair_n_z_style, self._opt.hair_num_clusters, 
                                        osp.join(root_path, self._opt.path_cluster_files, self._opt.hair_clusters), 
                                        self._opt.hair_b_min, self._opt.hair_b_max,
                                        self._opt.hair_resolution, thresh=self._opt.hair_thresh_occupancy)

        shoes = Model(osp.join(root_path, self._opt.path_checkpoints, 'shoes.pth'),
                                        self._opt.shoes_n_z_cut, 
                                        self._opt.shoes_n_z_style, self._opt.shoes_num_clusters, 
                                        osp.join(root_path, self._opt.path_cluster_files, self._opt.shoes_clusters), 
                                        self._opt.shoes_b_min, self._opt.shoes_b_max,
                                        self._opt.shoes_resolution, thresh=self._opt.shoes_thresh_occupancy)

        self.models = []
        for cloth_type in cloth_types:
            if cloth_type == 'uppercloth':
                self.models.append(uppercloth)
            elif cloth_type == 'coat':
                self.models.append(coat)
            elif cloth_type == 'pants':
                self.models.append(pants)
            elif cloth_type == 'skirts':
                self.models.append(skirts)
            elif cloth_type == 'hair':
                self.models.append(hair)
            elif cloth_type == 'shoes':
                self.models.append(shoes)
            else:
                assert 0, 'Not supported cloth type: ' + cloth_type
        self.cloth_types = cloth_types

        self.SMPL_Layers = [SMPL(osp.join(root_path, self._opt.path_SMPL, 'SMPL_NEUTRAL.pkl'), obj_saveable=True).cuda(),\
                            SMPL(osp.join(root_path, self._opt.path_SMPL, 'SMPL_MALE.pkl'), obj_saveable=True).cuda(),\
                            SMPL(osp.join(root_path, self._opt.path_SMPL, 'SMPL_FEMALE.pkl'), obj_saveable=True).cuda()]
        self.SMPL_Layer = None
        
        self.smpl_faces = self.SMPL_Layers[0].faces

        Astar_pose = torch.zeros(1, 72).cuda()
        Astar_pose[0, 5] = 0.04
        Astar_pose[0, 8] = -0.04
        self.register_buffer('Astar_pose', Astar_pose)

        # HYPERPARAMETER: Maximum number of points used when reposing.
        # This takes a lot of memory when finding the closest point in the SMPL so doing it by steps
        self.step = 1000 

    def get_right_shoe(self, sdf, unposed_cloth_mesh, do_marching_cube):
         
        # when not doing marching cube, mesh only contains vertices without faces
        if not do_marching_cube:
            sdf = torch.cat((sdf, sdf),1) # copy sdf
            rshoe = torch.stack((-unposed_cloth_mesh[:,:,0], unposed_cloth_mesh[:,:,1], unposed_cloth_mesh[:,:,2]),2)
            unposed_cloth_mesh = torch.cat((unposed_cloth_mesh, rshoe),1)
            return sdf, unposed_cloth_mesh
        # when doing marching cube, mesh contains both vertices and faces
        else:
            rshoe = np.stack((-unposed_cloth_mesh.vertices[:,0], unposed_cloth_mesh.vertices[:,1], unposed_cloth_mesh.vertices[:,2]),1)
            vertices = np.concatenate((unposed_cloth_mesh.vertices, rshoe))
            faces = np.concatenate((unposed_cloth_mesh.faces, unposed_cloth_mesh.faces[:,::-1] + len(rshoe)))
            unposed_cloth_mesh = trimesh.Trimesh(vertices, faces)
            return None, unposed_cloth_mesh

    def pose_mesh(self, unposed_cloth_mesh, pose, unposed_smpl_joint, unposed_smpl_mesh, do_marching_cube, smooth=True):
        if not do_marching_cube:
            iters = math.ceil(unposed_cloth_mesh.shape[1] / self.step)
            posed_cloth_mesh = []
            for i in range(iters):
                in_verts = unposed_cloth_mesh[:,i*self.step:(i+1)*self.step,:]
                out_verts = self.SMPL_Layer.deform_clothed_smpl(pose, unposed_smpl_joint, unposed_smpl_mesh, in_verts)
                posed_cloth_mesh.append(out_verts)
            posed_cloth_mesh = torch.cat(posed_cloth_mesh,1)
            return posed_cloth_mesh

        else:
            iters = math.ceil(len(unposed_cloth_mesh.vertices) / self.step)
            for i in range(iters):
                in_verts = torch.FloatTensor(unposed_cloth_mesh.vertices[None,i*self.step:(i+1)*self.step,:]).cuda()
                out_verts = self.SMPL_Layer.deform_clothed_smpl(pose, unposed_smpl_joint, unposed_smpl_mesh, in_verts)
                unposed_cloth_mesh.vertices[i*self.step:(i+1)*self.step] = out_verts.cpu().data.numpy() # replace unposed cloth mesh with posed one
            posed_cloth_mesh = unposed_cloth_mesh
            if smooth:
                posed_cloth_mesh = trimesh.smoothing.filter_laplacian(posed_cloth_mesh, lamb=0.5)
            return posed_cloth_mesh

    def pose_mesh_lower_body(self, unposed_cloth_mesh, pose, shape, Astar_pose, unposed_smpl_joint, unposed_smpl_mesh, do_marching_cube, smooth=True):
        if not do_marching_cube:
            iters = math.ceil(unposed_cloth_mesh.shape[1] / self.step)
            posed_cloth_mesh = []
            for i in range(iters):
                in_verts = unposed_cloth_mesh[:,i*self.step:(i+1)*self.step]
                out_verts = self.SMPL_Layer.unpose_and_deform_cloth(in_verts, Astar_pose, pose, shape, unposed_smpl_joint, unposed_smpl_mesh)
                posed_cloth_mesh.append(out_verts)
            posed_cloth_mesh = torch.cat(posed_cloth_mesh,1)
            return posed_cloth_mesh

        else:
            iters = math.ceil(len(unposed_cloth_mesh.vertices) / self.step)
            for i in range(iters):
                in_verts = torch.FloatTensor(unposed_cloth_mesh.vertices[None,i*self.step:(i+1)*self.step]).cuda()
                out_verts = self.SMPL_Layer.unpose_and_deform_cloth(in_verts, Astar_pose, pose, shape, unposed_smpl_joint, unposed_smpl_mesh)
                unposed_cloth_mesh.vertices[i*self.step:(i+1)*self.step] = out_verts.cpu().data.numpy() # replace unposed cloth mesh with posed one
            posed_cloth_mesh = unposed_cloth_mesh
            if smooth:
                posed_cloth_mesh = trimesh.smoothing.filter_laplacian(posed_cloth_mesh, lamb=0.5)
            return posed_cloth_mesh

    def forward(self, z_cuts, z_styles, pose, shape, gender=[0], do_marching_cube=False, valid=None, do_smooth=True):
        batch_size = pose.shape[0]
        
        unposed_smpl_joint, unposed_smpl_mesh = [], []
        Astar_smpl_mesh, Astar_smpl_joint = [], []
        for i in range(batch_size):
            SMPL_Layer = self.SMPL_Layers[gender[i]]
            unposed_smpl_joint_i, unposed_smpl_mesh_i = SMPL_Layer.skeleton(shape[None,i], require_body=True)
            Astar_smpl_mesh_i, Astar_smpl_joint_i, _ = SMPL_Layer.forward(beta=shape[None,i], theta=self.Astar_pose.repeat(1,1), get_skin=True)
            unposed_smpl_joint.append(unposed_smpl_joint_i); unposed_smpl_mesh.append(unposed_smpl_mesh_i)
            Astar_smpl_mesh.append(Astar_smpl_mesh_i); Astar_smpl_joint.append(Astar_smpl_joint_i)

        unposed_smpl_joint = torch.cat(unposed_smpl_joint); unposed_smpl_mesh = torch.cat(unposed_smpl_mesh)
        Astar_smpl_mesh = torch.cat(Astar_smpl_mesh); Astar_smpl_joint = torch.cat(Astar_smpl_joint)
        self.SMPL_Layer = self.SMPL_Layers[gender[0]]

        out_sdfs = []
        out_meshes = []
        out_meshes_unposed = []
        for i in range(len(self.models)):
            if ~valid[i]:
                out_sdfs.append([None])
                out_meshes.append([None])
                out_meshes_unposed.append([None])
                continue
            
            if self.cloth_types[i] in ['uppercloth', 'coat']:
                cloth_type = 'upperbody'
            else:
                cloth_type = self.cloth_types[i]
            resolution = eval(f'self._opt.{cloth_type}_resolution')

            if self.cloth_types[i] =='coat':
                is_coat = True
            else:
                is_coat = False
            
            if not do_marching_cube:
                resolution = 21

            if self.cloth_types[i] == 'pants' or self.cloth_types[i] == 'skirts':
                # forward network
                sdf, unposed_cloth_mesh = self.models[i].decode(z_cuts[i], z_styles[i], Astar_smpl_joint, Astar_smpl_mesh, resolution, do_marching_cube, do_smooth)

                # when not doing marching cube, all unposed_cloth_mesh have the same number of vertices
                if not do_marching_cube:
                    posed_cloth_mesh = self.pose_mesh_lower_body(unposed_cloth_mesh, pose, shape, self.Astar_pose.repeat(batch_size,1), unposed_smpl_joint, unposed_smpl_mesh, do_marching_cube)
                # when doing marching cube, unposed_cloth_mesh can have different number of vertices
                else:
                    posed_cloth_mesh = []
                    for j in range(len(unposed_cloth_mesh)):
                        if unposed_cloth_mesh[j] is None:
                            posed_cloth_mesh.append(None)
                            continue
                        posed_cloth_mesh.append(self.pose_mesh_lower_body(unposed_cloth_mesh[j], pose[j,None], shape[j,None], self.Astar_pose, unposed_smpl_joint[j,None], unposed_smpl_mesh[j,None], do_marching_cube, do_smooth))
            else:
                # forward network   
                sdf, unposed_cloth_mesh = self.models[i].decode(z_cuts[i], z_styles[i], unposed_smpl_joint, unposed_smpl_mesh, resolution, do_marching_cube, do_smooth, is_coat=is_coat)

                # when not doing marching cube, all unposed_cloth_mesh have the same number of vertices
                if not do_marching_cube:
                    if self.cloth_types[i] == 'shoes': # duplicate left shoe
                        sdf, unposed_cloth_mesh = self.get_right_shoe(sdf, unposed_cloth_mesh, do_marching_cube)
                    posed_cloth_mesh = self.pose_mesh(unposed_cloth_mesh, pose, unposed_smpl_joint, unposed_smpl_mesh, do_marching_cube)
                # when doing marching cube, unposed_cloth_mesh can have different number of vertices
                else:
                    posed_cloth_mesh = []
                    for j in range(len(unposed_cloth_mesh)):
                        if unposed_cloth_mesh[j] is None:
                            posed_cloth_mesh.append(None)
                            continue

                        if self.cloth_types[i] == 'shoes': # duplicate left shoe
                            _, unposed_cloth_mesh[j] = self.get_right_shoe(None, unposed_cloth_mesh[j], do_marching_cube)
                        posed_cloth_mesh.append(self.pose_mesh(unposed_cloth_mesh[j], pose[j,None], unposed_smpl_joint[j,None], unposed_smpl_mesh[j,None], do_marching_cube, do_smooth))

            out_sdfs.append(sdf)
            out_meshes.append(posed_cloth_mesh)
            out_meshes_unposed.append(unposed_cloth_mesh)
        
        return out_sdfs, out_meshes, out_meshes_unposed


