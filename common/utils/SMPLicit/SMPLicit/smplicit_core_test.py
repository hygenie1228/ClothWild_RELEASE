import torch
import numpy as np
from .utils.sdf import create_grid, eval_grid, eval_grid_octree
from skimage import measure
from .network import Network
import trimesh

class Model():
    def __init__(self, filename, n_z_cut, n_z_style, num_clusters, name_clusters, b_min, b_max, resolution, thresh=-0.05):
        self.filename = filename
        self.n_z_cut = n_z_cut
        self.n_z_style = n_z_style
        self.num_clusters = num_clusters
        self.clusters = np.load(name_clusters, allow_pickle=True)
        self.resolution = 128
        self.thresh = thresh
        self.load_networks()

    def load_networks(self):
        self._G = Network(n_z_style=self.n_z_style, point_pos_size=self.num_clusters*3, output_dim=1, n_z_cut=self.n_z_cut).cuda()
        self._G.load_state_dict(torch.load(self.filename))
        self._G.eval()

    def get_bbox(self, joint, mesh):
        joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')

        if 'upper' in self.filename:
            rhand = joint[:,joints_name.index('R_Hand'),:]
            lhand = joint[:,joints_name.index('L_Hand'),:]
            xmin = rhand[:,0]; xmax = lhand[:,0];
            ycenter = joint[:,joints_name.index('Chest'),1]
            height = (ycenter - joint[:,joints_name.index('Pelvis'),1])*2*2
            ymin = ycenter - height/2; ymax = ycenter + height/2;
            zcenter = (torch.min(mesh[:,:,2],1)[0] + torch.max(mesh[:,:,2],1)[0]) / 2.
            depth = (torch.max(mesh[:,:,2],1)[0] - torch.min(mesh[:,:,2],1)[0]) * 1.5
            zmin = zcenter - depth/2.; zmax = zcenter + depth/2.
            b_min = torch.stack((xmin, ymin, zmin),1)
            b_max = torch.stack((xmax, ymax, zmax),1)
            
        elif 'pants' in self.filename:
            rankle = joint[:,joints_name.index('R_Ankle'),:]
            lankle = joint[:,joints_name.index('L_Ankle'),:]
            pelvis = joint[:,joints_name.index('Pelvis'),:]
            spine1 = joint[:,joints_name.index('Torso'),:]
            xcenter = pelvis[:,0]; width = (xcenter - rankle[:,0])*2*2.3
            xmin = xcenter - width/2; xmax = xcenter + width/2;
            ycenter = (pelvis[:,1] + rankle[:,1])/2.; height = (pelvis[:,1] - ycenter)*2*1.2
            ymin = ycenter - height/2; ymax = ycenter + height/2;
            zcenter = (torch.min(mesh[:,:,2],1)[0] + torch.max(mesh[:,:,2],1)[0]) / 2.
            depth = (torch.max(mesh[:,:,2],1)[0] - torch.min(mesh[:,:,2],1)[0]) * 1.5
            zmin = zcenter - depth/2; zmax = zcenter + depth/2
            b_min = torch.stack((xmin, ymin, zmin),1)
            b_max = torch.stack((xmax, ymax, zmax),1)
            
        elif 'skirt' in self.filename:
            rankle = joint[:,joints_name.index('R_Ankle'),:]
            lankle = joint[:,joints_name.index('L_Ankle'),:]
            pelvis = joint[:,joints_name.index('Pelvis'),:]
            spine1 = joint[:,joints_name.index('Torso'),:]
            xcenter = pelvis[:,0]; width = (xcenter - rankle[:,0])*2*3
            xmin = xcenter - width/2; xmax = xcenter + width/2;
            ycenter = (pelvis[:,1] + rankle[:,1])/2.; height = (pelvis[:,1] - ycenter)*2*1.2
            ymin = ycenter - height/2; ymax = ycenter + height/2;
            zcenter = (torch.min(mesh[:,:,2],1)[0] + torch.max(mesh[:,:,2],1)[0]) / 2.
            depth = (torch.max(mesh[:,:,2],1)[0] - torch.min(mesh[:,:,2],1)[0]) * 2
            zmin = zcenter - depth/2; zmax = zcenter + depth/2
            b_min = torch.stack((xmin, ymin, zmin),1)
            b_max = torch.stack((xmax, ymax, zmax),1)
            
        elif 'hair' in self.filename:
            lshoulder = joint[:,joints_name.index('L_Shoulder'),:]
            rshoulder = joint[:,joints_name.index('R_Shoulder'),:]
            xcenter = (lshoulder[:,0] + rshoulder[:,0])/2.
            width = (xcenter - rshoulder[:,0])*2
            xmin = xcenter - width/2; xmax = xcenter + width/2;
            head = joint[:,joints_name.index('Head'),:]
            ymax = torch.max(mesh[:,:,1],1)[0]; ymin = joint[:,joints_name.index('Spine'),1]
            ycenter = (ymin+ymax)/2.; height = (ycenter - ymin)*2*1.2
            ymin = ycenter - height/2; ymax = ycenter + height/2;
            zcenter = (torch.min(mesh[:,:,2],1)[0] + torch.max(mesh[:,:,2],1)[0]) / 2.
            depth = (torch.max(mesh[:,:,2],1)[0] - torch.min(mesh[:,:,2],1)[0]) * 1.2
            zmin = zcenter - depth*0.8; zmax = zcenter + depth/2
            b_min = torch.stack((xmin, ymin, zmin),1)
            b_max = torch.stack((xmax, ymax, zmax),1)

        elif 'shoes' in self.filename:
            lknee = joint[:,joints_name.index('L_Knee'),:]
            lankle = joint[:,joints_name.index('L_Ankle'),:]
            lfoot = joint[:,joints_name.index('L_Toe'),:]
            xmin = lankle[:,0] - 0.15; xmax = lankle[:,0] + 0.15;
            ycenter = lankle[:,1]
            height = (lknee[:,1] - ycenter)*1.5
            ymin = ycenter - height/2; ymax = ycenter + height/2;
            zcenter = (lankle[:,2] + lfoot[:,2])/2.
            zmin = zcenter - 0.25; zmax = zcenter + 0.25;
            b_min = torch.stack((xmin, ymin, zmin),1)
            b_max = torch.stack((xmax, ymax, zmax),1)
            
        return b_min, b_max


    def decode(self, z_cut, z_style, smpl_joint, smpl_mesh, resolution, do_marching_cube, smooth=True, is_coat=False):
        batch_size = z_cut.shape[0]
        
        # prepare query points to predict SDF
        b_min, b_max = self.get_bbox(smpl_joint, smpl_mesh)
        query_points = create_grid((resolution, resolution, resolution), b_min, b_max)
        
        # sample points in clusters from smpl mesh
        smpl_points = smpl_mesh[:,self.clusters[self.num_clusters]] # batch_size, smpl_point_num, 3
        smpl_point_num = smpl_points.shape[1]
        
        def eval_func(query_points, ref_points, z_cut, z_style, scale):
            dist = query_points[:,:,None,:] - ref_points[:,None,:,:]
            dist = dist.view(-1, query_points.shape[1], ref_points.shape[1]*3)
            pred = self._G(z_cut, z_style, dist)*scale
            return pred
       
        if not do_marching_cube:
            if not smooth:
                # remove empty 3D space
                if 'upper' in self.filename:
                    query_points = query_points.view(batch_size, resolution, resolution, resolution, 3)
                    is_empty = torch.zeros((resolution, resolution, resolution)).float().cuda()
                    is_empty[:resolution//4,:resolution//2,:] = 1 # right
                    is_empty[resolution//4*3:,:resolution//2,:] = 1 # left
                    is_empty[resolution//4:resolution//4*3:,resolution//4:resolution//2,:resolution//3] = 1 # back center
                    query_points = query_points[is_empty[None,:,:,:,None].repeat(batch_size,1,1,1,3)==0].view(batch_size,-1,3)
                query_point_num = query_points.shape[1]

            # predict SDF
            sdf = eval_grid(query_points, smpl_points, z_cut, z_style, eval_func, resolution, 1, num_samples=10000)
            cloth_points = query_points 
            return sdf, cloth_points
            
        else:
            cloth_meshes = []
            sdfs = eval_grid(query_points, smpl_points, z_cut, z_style, eval_func, resolution, -100, num_samples=10000)
            sdfs = sdfs.view(batch_size,resolution,resolution,resolution)

            for i in range(batch_size):                
                sdf = sdfs[i].cpu().numpy()

                if 'pant' in self.filename:
                    # pant exception handling (heuristic)
                    sdf[resolution*63//128:resolution*66//128,:resolution*47//64,:] = self.thresh - 0.001
                    sdf[resolution*62//128:resolution*67//128,:resolution*45//64,:] = self.thresh - 0.001
                try:
                    verts, faces, normals, values = measure.marching_cubes(sdf, self.thresh, method='lewiner')

                    cloth_mesh = trimesh.Trimesh(np.float64(verts), faces[:, ::-1])
                    cloth_mesh.vertices /= resolution
                    cloth_mesh.vertices *= (b_max[i,None].cpu().numpy() - b_min[i,None].cpu().numpy())
                    cloth_mesh.vertices += b_min[i,None].cpu().numpy()

                    if smooth:
                        smooth_mesh = trimesh.smoothing.filter_laplacian(cloth_mesh, lamb=0.5)
                        if not np.isnan(smooth_mesh.vertices).any():
                            cloth_mesh = smooth_mesh
                    
                except ValueError:
                    cloth_mesh = None
                
                cloth_meshes.append(cloth_mesh)

            return None, cloth_meshes


