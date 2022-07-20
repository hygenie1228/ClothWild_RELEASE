import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, TexturesVertex
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.lighting import AmbientLights, PointLights
from pytorch3d.renderer.mesh.shader import BlendParams, HardPhongShader
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from config import cfg


def get_face_map(pix_to_face, faces):
    face_map = torch.zeros((pix_to_face.shape[0], pix_to_face.shape[1], 3)) - 1
    for i in range(pix_to_face.shape[0]):
        for j in range(pix_to_face.shape[1]):
            if pix_to_face[i][j] != -1:
                face_map[i][j] = faces[pix_to_face[i][j]]
    return face_map


class Renderer:
    def __init__(self, device='cuda', focal=cfg.focal, princpt=cfg.princpt, img_shape=cfg.input_img_shape):
        self.device = device

        self.set_renderer(focal, princpt, img_shape)

    def set_renderer(self, focal, princpt, img_shape, anti_aliasing=False):
        focal, princpt = torch.FloatTensor(focal)[None,:], torch.FloatTensor(princpt)[None,:]
        self.img_shape = img_shape
        self.anti_aliasing = anti_aliasing

        if self.anti_aliasing:
            img_shape = (img_shape[0]*2, img_shape[1]*2)
            princpt *= 2; focal *= 2

        img_size = max(img_shape[0], img_shape[1])
        raster_settings = RasterizationSettings(image_size=(img_size,img_size), blur_radius=0.0, faces_per_pixel=1, bin_size=0)

        cameras = PerspectiveCameras(focal_length=focal, \
                                        principal_point=princpt, \
                                        in_ndc=False, \
                                        R=torch.eye(3)[None,:,:], \
                                        T=torch.zeros(3)[None,:], \
                                        image_size=((img_size,img_size),),\
                                        device=torch.device(self.device))

        lights = PointLights(device=self.device, location=[[0.0, 0.0, -10.0]])
        materials = Materials(ambient_color=((0.92, 0.92, 0.92), ), diffuse_color=((1, 1, 1), ), specular_color=((1, 1, 1), ), shininess=4, device=self.device)
        blend_params = BlendParams(sigma=1e-1, gamma=1e-4)
        shader = HardPhongShader(device=self.device, blend_params=blend_params, cameras=cameras, lights=lights, materials=materials)

        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(self.device)
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=shader)

    def rasterize_mesh(self, mesh_vert, mesh_face):
        output = self.rasterizer(Meshes(verts=[mesh_vert.to(self.device)], faces=[mesh_face.to(self.device)]))

        face_map = get_face_map(output.pix_to_face.squeeze(), mesh_face)
        return face_map[:, :cfg.input_img_shape[1]]

    def render(self, img, mesh_vert, mesh_face):
        mesh_vert, mesh_face = torch.tensor(mesh_vert), torch.tensor(mesh_face)

        verts_rgb = torch.ones_like(mesh_vert)[None]
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        output = self.renderer(Meshes(verts=[mesh_vert.to(self.device)], faces=[mesh_face.to(self.device)], textures=textures))
        output = (output[0]*255).cpu().numpy()
        
        if self.anti_aliasing:
            img = cv2.resize(img, (self.img_shape[1]*2, self.img_shape[0]*2))
            img_shape = (self.img_shape[0]*2, self.img_shape[1]*2)
        else:
            img_shape = self.img_shape

        if img_shape[0] > img_shape[1]:
            output = output[:, :img_shape[1]]
        else:
            output = output[:img_shape[0], :]
        
        valid = output[:,:,3] > 0
        img[valid] = output[:,:,:3][valid]

        if self.anti_aliasing:
            img = cv2.resize(img, (self.img_shape[1], self.img_shape[0]))

        return img

def rasterize_mesh_given_cam_param(mesh_vert, mesh_face, focal, princpt):
    device = 'cuda'
    raster_settings = RasterizationSettings(image_size=(cfg.input_img_shape[0],cfg.input_img_shape[0]), blur_radius=0.0, faces_per_pixel=1)
    
    cameras = PerspectiveCameras(focal_length=torch.FloatTensor([focal[0],focal[1]])[None,:], \
                                    principal_point=torch.FloatTensor([princpt[0],princpt[1]])[None,:], \
                                    in_ndc=False, \
                                    R=torch.eye(3)[None,:,:], \
                                    T=torch.zeros(3)[None,:], \
                                    image_size=((cfg.input_img_shape[0],cfg.input_img_shape[0]),),\
                                    device=torch.device(device))
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    output = rasterizer(Meshes(verts=[mesh_vert.to(device)], faces=[mesh_face.to(device)]))

    face_map = get_face_map(output.pix_to_face.squeeze(), mesh_face)  
    return face_map[:, :cfg.input_img_shape[1]]

def save_proj_faces(face_map, save_path):
    face_map = face_map.reshape(-1, 3)

    file = open(save_path, 'w')
    for idx, v in enumerate(face_map):
        file.write('%d %d %d\n' % (v[0], v[1], v[2]))
    file.close()

def merge_mesh(verts, faces):
    vert_len = [0]
    for vert in verts:
        vert_len.append(len(vert))

    vert_len = np.cumsum(vert_len)
    for i, face in enumerate(faces):
        face += vert_len[i]
    
    return np.concatenate(verts), np.concatenate(faces)

def read_valid_point(verts, indexs, valid):
    valid_verts = []
    for i, val in enumerate(valid):
        if val != 0:
            idx1, idx2, idx3 = indexs[i]
            v = (verts[idx1] + verts[idx2] + verts[idx3]) / 3
            valid_verts.append(v)
    valid_verts = np.stack(valid_verts)
    return valid_verts
    
def pa_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    return a, R, t

def pairwise_distances(a, b, p=2, inv=False, num_samples=500):
    if not inv:
        tmp = a; a = b; b= tmp

    a = torch.tensor(a[None, :, :]).cuda()
    b = torch.tensor(b[None, :, :]).cuda()
    num_batches = a.shape[1] // num_samples

    dists = []
    for i in range(num_batches):
        dist = torch.norm((a[:,i*num_samples : (i+1)*num_samples, None, :] - b[:, None, :, :]),p=2,dim=3)
        dist, _ = torch.min(dist, 2)
        dist = dist.reshape(-1)
        dists.append(dist)

    if a.shape[1] % num_samples > 0:
        dist = torch.norm((a[:,-1 * (a.shape[1] % num_samples):, None, :] - b[:, None, :, :]),p=2,dim=3)
        dist, _ = torch.min(dist, 2)
        dist = dist.reshape(-1)
        dists.append(dist)

    dist= torch.cat(dists).mean().cpu()
    return dist

renderer = Renderer()