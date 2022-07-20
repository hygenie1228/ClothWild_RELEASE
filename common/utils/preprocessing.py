import numpy as np
import cv2
import random
from config import cfg
import math
from utils.human_models import smpl
from utils.transforms import cam2pixel, transform_joint_to_other_db
import torch

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid, extend_ratio=1.2):

    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, img_width, img_height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def convert_focal_princpt(focal, princpt, img2bb_trans):
    focal = np.array([[focal[0], 0], [0, focal[1]], [0, 0]])
    princpt = np.array([[princpt[0], 0], [0, princpt[1]], [1, 1]])

    focal = np.dot(img2bb_trans, focal)
    princpt = np.dot(img2bb_trans, princpt)

    cam_param = np.array([focal[0][0], focal[1][1], princpt[0][0], princpt[1][1]])
    return cam_param

def get_aug_config():
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    do_flip = False
    return scale, rot, color_scale, do_flip

def augmentation(img, bbox, data_split):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config()
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False
    
    img, valid_mask, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    return img, valid_mask, trans, inv_trans, rot, do_flip

def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR, borderValue=(-1,-1,-1))
    valid_mask = (img_patch > -1)
    if len(valid_mask.shape) == 3:
        valid_mask = valid_mask[:,:,0]
    img_patch[img_patch == -1] = 0
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, valid_mask, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def process_db_coord(joint_img, joint_valid, do_flip, img_shape, flip_pairs, img2bb_trans, rot, src_joints_name, target_joints_name):
    joint_img, joint_valid = joint_img.copy(), joint_valid.copy()

    # flip augmentation
    if do_flip:
        joint_img[:,0] = img_shape[1] - 1 - joint_img[:,0]
        for pair in flip_pairs:
            joint_img[pair[0],:], joint_img[pair[1],:] = joint_img[pair[1],:].copy(), joint_img[pair[0],:].copy()
            joint_valid[pair[0],:], joint_valid[pair[1],:] = joint_valid[pair[1],:].copy(), joint_valid[pair[0],:].copy()
    
    # affine transformation and root-relative depth
    joint_img_xy1 = np.concatenate((joint_img, np.ones_like(joint_img[:,:1])),1)
    joint_img = np.dot(img2bb_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
    joint_img[:,0] = joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_joint_shape[1]
    joint_img[:,1] = joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_joint_shape[0]
    
    # check truncation
    joint_trunc = joint_valid * ((joint_img[:,0] >= 0) * (joint_img[:,0] < cfg.output_joint_shape[1]) * \
                (joint_img[:,1] >= 0) * (joint_img[:,1] < cfg.output_joint_shape[0])).reshape(-1,1).astype(np.float32)


    # transform joints to target db joints
    joint_img = transform_joint_to_other_db(joint_img, src_joints_name, target_joints_name)
    joint_valid = transform_joint_to_other_db(joint_valid, src_joints_name, target_joints_name)
    joint_trunc = transform_joint_to_other_db(joint_trunc, src_joints_name, target_joints_name)
    return joint_img, joint_valid, joint_trunc

def process_human_model_output(human_model_param, cam_param, do_flip, img_shape, img2bb_trans, rot):
    pose, shape = human_model_param['pose'], human_model_param['shape']

    if 'trans' in human_model_param:
        trans = human_model_param['trans']
    else:
        trans = [0,0,0]
    if 'gender' in human_model_param:
        gender = human_model_param['gender']
    else:
        gender = 'neutral'
    pose = torch.FloatTensor(pose).view(-1,3); shape = torch.FloatTensor(shape).view(1,-1); # smpl parameters (pose: 72 dimension, shape: 10 dimension)
    trans = torch.FloatTensor(trans).view(1,-1) # translation vector
    
    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation 
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
        root_pose = pose[smpl.root_joint_idx,:].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
        pose[smpl.root_joint_idx] = torch.from_numpy(root_pose).view(3)

    # get mesh and joint coordinates
    root_pose = pose[smpl.root_joint_idx].view(1,3)
    body_pose = torch.cat((pose[:smpl.root_joint_idx,:], pose[smpl.root_joint_idx+1:,:])).view(1,-1)
    output = smpl.layer[gender](betas=shape, body_pose=body_pose, global_orient=root_pose, transl=trans)
    mesh_coord = output.vertices[0].numpy()
    joint_coord = np.dot(smpl.joint_regressor, mesh_coord)

    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)
    if 'R' in cam_param and 't' in cam_param:
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
        root_coord = joint_coord[smpl.root_joint_idx,None,:]
        joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
        mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t

    ## so far, data augmentations are not applied yet
    ## now, project the 3D coordinates to image space and apply data augmentations

    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)
    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in smpl.flip_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
        pose[:,1:3] *= -1 # multiply -1 to y and z axis of axis-angle
    # rotate root pose
    pose = pose.numpy()
    root_pose = pose[smpl.root_joint_idx,:]
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
    pose[smpl.root_joint_idx] = root_pose.reshape(3)
    
    # return results
    pose = pose.reshape(-1)
    # change to mean shape if beta is too far from it
    shape[(shape.abs() > 3).any(dim=1)] = 0.
    shape = shape.numpy().reshape(-1)
    return pose, shape, mesh_coord # data augmentation is not performed on mesh_coord 

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[2]-1);
    x1 = np.clip(x1, 0, im.shape[2]-1);
    y0 = np.clip(y0, 0, im.shape[1]-1);
    y1 = np.clip(y1, 0, im.shape[1]-1);

    Ia = im[:, y0, x0 ]
    Ib = im[:, y1, x0 ]
    Ic = im[:, y0, x1 ]
    Id = im[:, y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def iou_sil(sil_out, sil_target):
    intersect = sil_out * sil_target
    union = (sil_out + sil_target) > 0
    if np.sum(union) == 0:
        return None
    else:
        return np.sum(intersect) / np.sum(union)
