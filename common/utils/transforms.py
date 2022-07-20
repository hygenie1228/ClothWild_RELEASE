import torch
import numpy as np
from config import cfg
from torch.nn import functional as F
import torchgeometry as tgm

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1) # 3x3 rotation matrix
    
    rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).cuda().float()],2) # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def unwrap_xy_to_uv(feat_xy, dp_fg, dp_I, dp_u, dp_v):
    batch_size, feat_dim, height, width = feat_xy.shape

    dp_fg = torch.max(dp_fg, 1)[1] # argmax
    dp_I = torch.max(dp_I, 1)[1] + 1 # argmax. add 1 to make the bkg class
    dp_I[dp_fg == 0] = 0 # bkg

    _dp_u, _dp_v = 0, 0
    for i in range(cfg.dp_patch_num):
        mask = (dp_I == (i+1)) # add 1 to make the bkg class
        _dp_u += dp_u[:,i,:,:] * mask
        _dp_v += dp_v[:,i,:,:] * mask
    dp_u, dp_v = _dp_u, _dp_v

    
    scatter_src = feat_xy.permute(1,0,2,3).reshape(feat_dim,-1)
    
    batch_idx = torch.arange(batch_size)[:,None,None].repeat(1,height,width).view(-1).to(feat_xy.device) #.cuda()
    _dp_I = dp_I.view(-1)
    _dp_u = (dp_u.view(-1) * (cfg.output_uv_shape[0]-1)).long()
    _dp_v = ((1 - dp_v.view(-1)) * (cfg.output_uv_shape[1]-1)).long() # inverse v coordinate following DensePose R-CNN
    scatter_idx = batch_idx * (cfg.dp_patch_num + 1) * cfg.output_uv_shape[0] * cfg.output_uv_shape[1] + \
            _dp_I * cfg.output_uv_shape[0] * cfg.output_uv_shape[1] + \
            _dp_u * cfg.output_uv_shape[1] + \
            _dp_v

    is_valid = (_dp_u >= 0) * (_dp_u < cfg.output_uv_shape[0]) * (_dp_v >= 0) * (_dp_v < cfg.output_uv_shape[1])
    scatter_src = scatter_src[:,is_valid]
    scatter_idx = scatter_idx[is_valid]
    
    feat_uv = scatter_mean(scatter_src, scatter_idx, 1, dim_size = batch_size * (cfg.dp_patch_num + 1) * cfg.output_uv_shape[0] * cfg.output_uv_shape[1]).view(feat_dim, batch_size, cfg.dp_patch_num + 1, cfg.output_uv_shape[0], cfg.output_uv_shape[1]).permute(1,2,0,3,4)[:,1:,:,:,:] # remove bkg class (cfg.dp_patch_num + 1 -> cfg.dp_patch_num)

    return feat_uv

