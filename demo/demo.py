import sys
import os
import os.path as osp

sys.path.insert(0, osp.join('..', 'main'))
from config import cfg

import argparse
import json
import torch
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.parallel.data_parallel import DataParallel
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--img_path', type=str, default='./input.jpg')
    parser.add_argument('--json_path', type=str, default='./pose2pose_result.json')
    parser.add_argument('--output_dir', type=str, default='./')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

from model import get_model
from base import check_data_parallel
from utils.preprocessing import load_img, process_bbox, generate_patch_image, process_human_model_output
from utils.postprocessing import renderer
from utils.vis import save_result, render_result

model_path = os.path.join('.', 'snapshot_7.pth.tar')
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

model = get_model('test')
model = model.cuda()
ckpt = torch.load(model_path)
ckpt = check_data_parallel(ckpt['network'])
model.load_state_dict(ckpt, strict=False)
model.eval()

transform = transforms.ToTensor()    
original_img = load_img(args.img_path)
original_height, original_width = original_img.shape[:2]

with open(args.json_path, 'r') as f:
    pose2pose_result = json.load(f)

# prepare bbox
bbox = [150, 38, 244, 559]
bbox = process_bbox(bbox, original_width, original_height)
img_numpy, _, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
img = transform(img_numpy.astype(np.float32))/255.
img = img.cuda()[None,:,:,:]

smpl_pose, smpl_shape, smpl_mesh = process_human_model_output(pose2pose_result['smpl_param'], pose2pose_result['cam_param'], False, (original_width, original_height), img2bb_trans, 0.0)
cam_trans = np.array(pose2pose_result['smpl_param']['trans'], dtype=np.float32)
smpl_pose, smpl_shape, cam_trans = torch.tensor(smpl_pose)[None,:].cuda(), torch.tensor(smpl_shape)[None,:].cuda(), torch.tensor(cam_trans)[None,:].cuda()

# forward
inputs = {'img': img}
targets = {}
meta_info = {'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape, 'cam_trans': cam_trans}

with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')

for k,v in out.items():
    if type(v) is torch.Tensor:
        out[k] = v[0].cpu().numpy()
    else:
        out[k] = v[0]

mesh_verts, mesh_faces = save_result(out, osp.join(args.output_dir, 'output.obj'))
renderer.set_renderer(focal=cfg.focal, princpt=cfg.princpt, img_shape=cfg.input_img_shape, anti_aliasing=True)
render_result(mesh_verts, mesh_faces, img_numpy[:,:,::-1], osp.join(args.output_dir, 'render_cropped_img.jpg'))

focal = [cfg.focal[0] / cfg.input_img_shape[1] * bbox[2], cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]]
princpt = [cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]]
renderer.set_renderer(focal=focal, princpt=princpt, img_shape=(original_height, original_width), anti_aliasing=True)
render_result(mesh_verts, mesh_faces, original_img[:,:,::-1], osp.join(args.output_dir, 'render_original_img.jpg'))