import numpy as np
import torch
import math

def create_grid(resolution, b_min, b_max):
    batch_size = b_min.shape[0]

    # make grids
    res_x, res_y, res_z = resolution
    #zz,yy,xx = torch.meshgrid(torch.arange(res_z),torch.arange(res_y),torch.arange(res_x))
    xx,yy,zz = torch.meshgrid(torch.arange(res_x),torch.arange(res_y),torch.arange(res_z))
    coords = torch.stack((xx, yy, zz))
    coords = coords.reshape(3, -1).float()
    coords = coords[None,:,:].repeat(batch_size,1,1).float().cuda()
    

    # affine transform
    coords_matrix = torch.eye(4).view(1,4,4).repeat(batch_size,1,1).float().cuda()
    length = b_max - b_min
    coords_matrix[:, 0, 0] = length[:,0] / res_x
    coords_matrix[:, 1, 1] = length[:,1] / res_y
    coords_matrix[:, 2, 2] = length[:,2] / res_z
    coords_matrix[:, 0:3, 3] = b_min
    coords = torch.bmm(coords_matrix[:, :3, :3], coords) + coords_matrix[:, :3, 3:4]

    # return grids
    coords = coords.view(batch_size, 3, -1).transpose(2,1).contiguous() # res_x*res_y*res_z, 3
    return coords

def batch_eval(query_points, ref_points, z_cut, z_style, eval_func, scale, num_samples):
    num_pts = query_points.shape[1]
    num_batches = math.ceil(num_pts / num_samples)
    
    sdf = []
    for i in range(num_batches):
        sdf.append(eval_func(query_points[:,i * num_samples:i * num_samples + num_samples,:], ref_points, z_cut, z_style, scale))
    sdf = torch.cat(sdf,1)
    return sdf

def eval_grid(query_points, ref_points, z_cut, z_style, eval_func, resolution, scale, num_samples=512 * 512 * 512):
    sdf = batch_eval(query_points, ref_points, z_cut, z_style, eval_func, scale, num_samples=num_samples)
    return sdf

def eval_grid_octree(query_points, ref_points, z_cut, z_style, eval_func, resolution, init_resolution=64, threshold=0.01, num_samples=512 * 512 * 512):
    res_x, res_y, res_z = resolution
    sdf = np.zeros(resolution)
    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    step_size = res_x // init_resolution
    while step_size > 0:
        # subdivide the grid
        grid_mask[0:res_x:step_size, 0:res_y:step_size, 0:res_z:step_size] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        points = query_points[torch.from_numpy(test_mask).cuda().reshape(-1)==1,:]
        
        sdf[test_mask] = batch_eval(points[None,:,:], ref_points[None,:,:], z_cut[None,:], z_style[None,:], eval_func, num_samples=num_samples).detach().cpu().numpy().reshape(-1)
        dirty[test_mask] = False

        # do interpolation
        if step_size <= 1:
            break
        for x in range(0, res_x - step_size, step_size):
            for y in range(0, res_y - step_size, step_size):
                for z in range(0, res_z - step_size, step_size):
                    # if center marked, return
                    if not dirty[x + step_size // 2, y + step_size // 2, z + step_size // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + step_size]
                    v2 = sdf[x, y + step_size, z]
                    v3 = sdf[x, y + step_size, z + step_size]
                    v4 = sdf[x + step_size, y, z]
                    v5 = sdf[x + step_size, y, z + step_size]
                    v6 = sdf[x + step_size, y + step_size, z]
                    v7 = sdf[x + step_size, y + step_size, z + step_size]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # this cell is all the same
                    if (v_max - v_min) < threshold:
                        sdf[x:x + step_size, y:y + step_size, z:z + step_size] = (v_max + v_min) / 2
                        dirty[x:x + step_size, y:y + step_size, z:z + step_size] = False
        step_size //= 2

    return sdf.reshape(resolution)

