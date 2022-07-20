import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import pickle as pkl
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import smpl
from utils.preprocessing import load_img, process_bbox, augmentation, process_human_model_output, convert_focal_princpt
from utils.postprocessing import renderer, rasterize_mesh_given_cam_param, save_proj_faces, merge_mesh, read_valid_point, pa_mpjpe, pairwise_distances
from utils.vis import save_obj, save_result, render_result

class PW3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'PW3D', 'data')
        self.sequence_path = osp.join('..', 'data', 'PW3D', 'data', 'sequenceFiles', self.data_split)
        self.preprocessed_path = osp.join('..', 'data', 'preprocessed_data')
        self.eval_stride = 25
        self.cd_inlier_threshold = 32

        self.pw3d_smpl_layers = {}
        self.pw3d_beta_clothes = {}

        self.datalist = self.load_data()
        print(f"Load {self.data_split} data: ", len(self.datalist))

    def load_data(self):
        db = COCO(osp.join(self.data_path, '3DPW_' + self.data_split + '.json'))

        if self.data_split == 'test':
            with open(osp.join(self.preprocessed_path, 'smpl_param', '3DPW_test_Pose2Pose.json')) as f:
                smpl_params = json.load(f)
        else:
            smpl_params = None

        datalist = []
        for idx, aid in enumerate(db.anns.keys()):
            if idx % self.eval_stride != 0: continue

            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            sequence_name = img['sequence']
            img_name = img['file_name']
            pid = ann['person_id']
            img_path = osp.join(self.data_path, 'imageFiles', sequence_name, img_name)
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue

            cam_param_gt = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
            smpl_param_gt = ann['smpl_param']
            gender = smpl_param_gt['gender']

            if str(aid) in smpl_params:
                smpl_param = smpl_params[str(aid)]['smpl_param']
                cam_param  = smpl_params[str(aid)]['cam_param']
            else:
                assert 0, "SMPL params missed!"
            
            # pre-save smpl layers
            if cfg.calculate_cd:
                sequence =  img_path.split('/')[-2]
                index = sequence + '_' + str(pid)
                if index not in self.pw3d_smpl_layers.keys():
                    data = pkl.load(open(osp.join(self.sequence_path, f'{sequence}.pkl'), 'rb'), encoding='latin1')
                    v_template = data['v_template_clothed'][pid]
                    betas_clothed = data['betas_clothed'][pid][:10]
                    layer = smpl.get_custom_template_layer(v_template, gender)
                    self.pw3d_smpl_layers[index] = layer
                    self.pw3d_beta_clothes[index] = betas_clothed

            data_dict = {'img_path': img_path, 'ann_id': aid, 'person_id': pid, 'img_shape': (img['height'], img['width']), 'bbox': bbox, 
                        'smpl_param': smpl_param, 'cam_param': cam_param, 'smpl_param_gt': smpl_param_gt, 'cam_param_gt': cam_param_gt}

            datalist.append(data_dict)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape = data['img_path'], data['img_shape']
        # image load
        img = load_img(img_path)

        # affine transform
        bbox = data['bbox']
        img, valid_mask, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        # smpl processing
        smpl_pose, smpl_shape, smpl_mesh = process_human_model_output(data['smpl_param'], data['cam_param'], do_flip, img_shape, img2bb_trans, rot)
        cam_trans = np.array(data['smpl_param']['trans'], dtype=np.float32)
        cam_param = np.array([cfg.focal[0], cfg.focal[1], cfg.princpt[0], cfg.princpt[1]])

        inputs = {'img': img}
        targets = {}
        meta_info = {'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape, 'cam_trans': cam_trans, 'cam_param': cam_param}

        if cfg.calculate_cd:
            smpl_pose, smpl_shape, cam_trans, gender = data['smpl_param_gt']['pose'], data['smpl_param_gt']['shape'], data['smpl_param_gt']['trans'], data['smpl_param_gt']['gender']
            smpl_pose, smpl_shape, cam_trans = np.array(smpl_pose), np.array(smpl_shape), np.array(cam_trans)
            cam_param = convert_focal_princpt(data['cam_param_gt']['focal'], data['cam_param_gt']['princpt'], img2bb_trans)

            smpl_mesh = self.get_clothed_mesh(img_path.split('/')[-2], data['person_id'], smpl_pose, smpl_shape, cam_trans, gender)
            targets['smpl_mesh'] = smpl_mesh
            targets['cam_param'] = cam_param
        
        return inputs, targets, meta_info
    
    def get_clothed_mesh(self, sequence, pid, pose, shape, trans, gender):
        index = sequence + '_' + str(pid)
        layer = self.pw3d_smpl_layers[index]
        betas_clothed = self.pw3d_beta_clothes[index]

        pose = torch.FloatTensor(pose).view(1,-1); shape = torch.FloatTensor(betas_clothed).view(1,-1); trans = torch.FloatTensor(trans).view(1,-1)
        output = layer(betas=shape, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=trans)
        mesh_cam = output.vertices[0].numpy()

        output = smpl.layer['neutral'](betas=shape, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=trans)
        unclothed_mesh_cam = output.vertices[0].numpy()

        trans = np.mean(mesh_cam, axis=0) - np.mean(unclothed_mesh_cam, axis=0)
        mesh_cam -= trans
        return mesh_cam

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        start_idx = 0

        eval_result = {'chamfer_distance': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            ann_id = annot['ann_id']
            out = outs[n]

            # save ouputs for calcuting cd
            if cfg.calculate_cd:
                verts_out = []; faces_out = []
                verts = out['smpl_mesh']
                verts[:,:2] *= -1
                verts_out.append(verts)
                faces_out.append(smpl.face.astype(np.int32))

                for cloth_type in cfg.cloth_types:
                    if out[cloth_type + '_mesh'] is None: continue
                    verts = out[cloth_type + '_mesh'].vertices
                    verts[:,:2] *= -1
                    verts_out.append(verts)
                    faces_out.append(out[cloth_type + '_mesh'].faces.astype(np.int32))
                
                # pred
                pred_verts, pred_faces = merge_mesh(verts_out, faces_out)
                pred_faces = renderer.rasterize_mesh(torch.from_numpy(pred_verts).float(), torch.from_numpy(pred_faces))

                # gt
                gt_verts = out['smpl_mesh_target']; gt_faces = smpl.face.astype(np.int32)
                gt_verts[:,:2] *= -1
                gt_faces = rasterize_mesh_given_cam_param(torch.from_numpy(gt_verts).float(), torch.from_numpy(gt_faces), out['cam_param_target'][:2], out['cam_param_target'][2:])

                # find valid pixels - exist silhouette
                pred_faces, gt_faces = pred_faces.numpy().astype(np.int32).reshape(-1, 3), gt_faces.numpy().astype(np.int32).reshape(-1, 3)
                valid = (pred_faces!=-1).sum(1) * (gt_faces!=-1).sum(1)
                valid = valid.reshape(-1)

                # if there are too few valid points, not evaluate
                if valid.sum() < self.cd_inlier_threshold:
                    continue

                # set semantically matching pairs
                paired_pred_verts = read_valid_point(pred_verts, pred_faces, valid)
                paired_gt_verts = read_valid_point(gt_verts, gt_faces, valid)

                # rigid alignment
                a, R, t = pa_mpjpe(np.expand_dims(paired_pred_verts,0), np.expand_dims(paired_gt_verts,0))
                pred_verts = (a*np.matmul(pred_verts, R) + t)[0]
                pred_verts *= 1000; gt_verts *= 1000

                # pcu.pairwise_distances is too slow, approximate distance between vertices.
                dist1 = pairwise_distances(pred_verts, gt_verts)
                dist2 = pairwise_distances(pred_verts, gt_verts, inv=True)

                if torch.isinf(dist1) or torch.isinf(dist2): continue

                chamfer_dist = (dist1 + dist2) / 2
                eval_result['chamfer_distance'].append(chamfer_dist)
        
        return eval_result

    def print_eval_result(self, eval_result):
        print('>> CD: %.2f mm' % np.mean(eval_result['chamfer_distance']))

                
