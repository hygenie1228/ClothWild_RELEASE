import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from utils.human_models import smpl
from utils.preprocessing import load_img, process_bbox, augmentation, generate_patch_image, process_db_coord, process_human_model_output, bilinear_interpolate, iou_sil
from utils.vis import save_obj, save_result, render_result

class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'MSCOCO', 'images')
        self.annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations')
        self.parse_path = osp.join('..', 'data', 'MSCOCO', 'parses')
        self.preprocessed_path = osp.join('..', 'data', 'preprocessed_data')
        self.dp_path = osp.join(self.preprocessed_path, 'densepose', 'MSCOCO')

        # lip parse set
        self.parse_set = {'uppercloth': (5,), 'coat': (7,), 'pants': (9,), 'skirts': (12,), 'hair': (2,), 'shoes': (18,19)}
        self.bcc_dist_threshold = 0.03
        self.eval_types = ['upper_body', 'lower_body','non_cloth']

        self.datalist = self.load_data()
        print("Load data: ", len(self.datalist))

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_train_v1.0.json'))
            with open(osp.join(self.preprocessed_path, 'gender', 'MSCOCO_train_gender.json')) as f:
                genders = json.load(f)
            with open(osp.join(self.parse_path, 'LIP_trainval_parsing.json')) as f:
                parsing_paths = json.load(f)
        else:
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_val_v1.0.json'))
            with open(osp.join(self.preprocessed_path, 'gender', 'MSCOCO_val_gender.json')) as f:
                genders = json.load(f)
            with open(osp.join(self.preprocessed_path, 'smpl_param', 'MSCOCO_test_Pose2Pose.json')) as f:
                smpl_params = json.load(f)
            if cfg.calculate_bcc:
                with open(osp.join(self.parse_path, 'LIP_trainval_parsing.json')) as f:
                    parsing_paths = json.load(f)
                with open(osp.join(self.annot_path, 'coco_dp_val.json')) as f:
                    dps = json.load(f)
            else:
                parsing_paths = None
                dps = None

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]

            if self.data_split == 'train':
                imgname = osp.join('train2017', img['file_name'])
            else:
                imgname = osp.join('val2017', img['file_name'])
            img_path = osp.join(self.img_path, imgname)
            
            if self.data_split == 'train':
                if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue
            
            # bbox
            bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
            if bbox is None: continue
        
            # parse
            if parsing_paths is not None:
                if str(aid) in parsing_paths:
                    parse_path = osp.join(self.parse_path, 'TrainVal_parsing_annotations/TrainVal_parsing_annotations/', parsing_paths[str(aid)])
                else:
                    continue
            else:
                parse_path = None

            # filter images with few visible joints
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
            if np.sum(joint_img[:,2]>0) < 6: continue

            # densepose
            if self.data_split == 'train':
                try:
                    dp = np.load(osp.join(self.dp_path, str(aid) + '.npz'), allow_pickle=True)
                except:
                    continue

                if len(dp['smpl_v_idx']) == 0: continue

                dp_x = np.array(dp['dp_x'], dtype=np.float32)
                dp_y = np.array(dp['dp_y'], dtype=np.float32)
                dp_xy = np.stack((dp_x, dp_y),1)

                dp_I = np.array(dp['dp_I'], dtype=np.int16)
                dp_u = np.array(dp['dp_U'], dtype=np.float32)
                dp_v = np.array(dp['dp_V'], dtype=np.float32)
                dp_uv = np.stack((dp_u, dp_v),1)
                
                smpl_v_idx = np.array(dp['smpl_v_idx'], dtype=np.int32)
                dp_mask = dp['dp_fg'].item()
                dp_mask = mask_util.decode(dp_mask)

                dp_data = {'xy': dp_xy, 'uv': dp_uv, 'I': dp_I, 'smpl_v_idx': smpl_v_idx, 'masks': dp_mask}
            elif cfg.calculate_bcc:
                if str(aid) in dps:
                    dp = dps[str(aid)]
                    if len(dp['smpl_v_idx']) == 0: continue

                    dp_x = np.array(dp['dp_x'], dtype=np.float32) / 256 * ann['bbox'][2] + ann['bbox'][0]
                    dp_y = np.array(dp['dp_y'], dtype=np.float32) / 256 * ann['bbox'][3] + ann['bbox'][1]
                    dp_xy = np.stack((dp_x, dp_y),1)

                    dp_I = np.array(dp['dp_I'], dtype=np.int16)
                    dp_u = np.array(dp['dp_U'], dtype=np.float32)
                    dp_v = np.array(dp['dp_V'], dtype=np.float32)
                    dp_uv = np.stack((dp_u, dp_v),1)
                    
                    smpl_v_idx = np.array(dp['smpl_v_idx'], dtype=np.int32)
                    dp_mask = mask_util.decode(dp['dp_masks'][0])

                    dp_data = {'xy': dp_xy, 'uv': dp_uv, 'I': dp_I, 'smpl_v_idx': smpl_v_idx, 'masks': dp_mask}
                else:
                    continue
            else:
                dp_data = None

            # smpl params
            if self.data_split == 'test':
                if str(aid) in smpl_params:
                    smpl_param = smpl_params[str(aid)]['smpl_param']
                    cam_param  = smpl_params[str(aid)]['cam_param']
                else:
                    continue
            else:
                smpl_param, cam_param = None, None

            # gender
            if str(aid) in genders:
                gender = genders[str(aid)]
            else:
                continue

            data_dict = {
                'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 
                'bbox': bbox, 'orig_bbox': ann['bbox'], 'gender': gender, 
                'parse_path': parse_path, 'dp': dp_data,
                'smpl_param': smpl_param, 'cam_param': cam_param
                } 
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

        if self.data_split == 'train':
            # load parse (cloth segmentation)
            parse = cv2.imread(data['parse_path'])
            parse_list = []
            for cloth_type in ('fg',) + cfg.cloth_types:
                # get cloth indexs
                if cloth_type == 'fg':
                    idxs = np.unique(parse).tolist()
                    idxs.pop(idxs.index(0))
                    if len(idxs) == 0:
                        parse_fg = np.zeros((cfg.output_parse_shape[0], cfg.output_parse_shape[1])) > 0
                        continue
                else:
                    idxs = self.parse_set[cloth_type]
                
                # get masking corresponding to a cloth
                mask = [parse == i for i in idxs]
                mask = (sum(mask) > 0).astype(np.float32)
                _, _, _, lip2img_trans = generate_patch_image(mask, data['orig_bbox'], 1.0, 0.0, False, mask.shape)
                mask = cv2.warpAffine(mask, lip2img_trans, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(mask, img2bb_trans, (cfg.input_img_shape[1], cfg.input_img_shape[0]), flags=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (cfg.output_parse_shape[1], cfg.output_parse_shape[0]))
                
                if cloth_type == 'fg': parse_fg = mask[:,:,0] > 0
                else: parse_list.append(mask)

            parse = np.stack(parse_list)[:,:,:,0] # parse: (cloths, height, width)
            is_bkg = (np.prod(parse == 0, 0) == 1)
            parse = np.argmax(parse, 0) + 1 # add 1 for bkg class
            parse[is_bkg] = 0
            parse[valid_mask == 0 ] = -1
            parse_valid = valid_mask

            # load densepose
            dp_xy, dp_uv, dp_I, dp_vertex = data['dp']['xy'], data['dp']['uv'], data['dp']['I'], data['dp']['smpl_v_idx']
            dp_xy = np.concatenate((dp_xy, np.ones_like(dp_xy[:,:1])),1)
            dp_xy = np.dot(img2bb_trans, dp_xy.transpose(1,0)).transpose(1,0)
            dp_I = dp_I - 1 # dp_I is started wtih 1. make it zero-based index.

            cur_point_num = len(dp_xy)
            if cur_point_num > cfg.dp_point_num:
                idxs = np.random.choice(np.arange(cur_point_num), size=cfg.dp_point_num)
                cur_point_num = cfg.dp_point_num
                dp_xy = dp_xy[idxs]; dp_uv = dp_uv[idxs]; dp_I = dp_I[idxs]; dp_vertex = dp_vertex[idxs]
            
            # match densepose & parse
            _dp_xy = dp_xy.copy()
            _dp_xy[:,0] = _dp_xy[:,0] / cfg.input_img_shape[1] * cfg.output_parse_shape[1]
            _dp_xy[:,1] = _dp_xy[:,1] / cfg.input_img_shape[0] * cfg.output_parse_shape[0]

            parse_onehot = np.zeros((len(cfg.cloth_types)+1, cfg.output_parse_shape[0], cfg.output_parse_shape[1]))
            for i in range(len(cfg.cloth_types)+1):
                parse_onehot[i][parse == i] = 1.0

            dp_cloth_idx = np.ones((_dp_xy.shape[0]), np.int16) * -1
            dp_cloth_idx[bilinear_interpolate(parse_fg[None,:,:], _dp_xy[:,0], _dp_xy[:,1])[0] > 0.5] = 0
            for i in range(len(cfg.cloth_types)):
                dp_cloth_idx[np.argmax(bilinear_interpolate(parse_onehot, _dp_xy[:,0], _dp_xy[:,1]), 0) == (i+1)] = i+1

            smpl_cloth_idx = np.ones((smpl.vertex_num), dtype=np.int16) * -1
            smpl_cloth_idx[dp_vertex] = dp_cloth_idx
            smpl_cloth_valid = (smpl_cloth_idx != -1).astype(np.float32)
            smpl_patch_idx = np.ones((smpl.vertex_num), dtype=np.int16) * -1
            smpl_patch_idx[dp_vertex] = dp_I[dp_I != -1]

            # remove coat ambiguity
            if (smpl_cloth_idx==cfg.cloth_types.index('uppercloth')+1).sum() == 0 and (smpl_cloth_idx==cfg.cloth_types.index('coat')+1).sum() > 0:
                idxs = (smpl_cloth_idx == cfg.cloth_types.index('coat')+1)
                smpl_cloth_idx[idxs] = cfg.cloth_types.index('uppercloth')+1
            
            # gender
            if data['gender'] == 'male': gender = 1
            elif data['gender'] == 'female': gender = 2
            else: gender = 0

            # dummy smpl parameter
            smpl_pose = np.zeros((smpl.joint_num*3,), dtype=np.float32)
            smpl_shape = np.zeros((smpl.shape_param_dim,), dtype=np.float32)
            cam_trans = np.zeros((3,), dtype=np.float32)

            inputs = {'img': img}
            targets = {'gender': gender, 'parse': parse, 'smpl_cloth_idx': smpl_cloth_idx, 'smpl_patch_idx': smpl_patch_idx}
            meta_info = {'smpl_cloth_valid': smpl_cloth_valid, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape, 'cam_trans': cam_trans}
            return inputs, targets, meta_info

        else:
            # smpl processing
            smpl_pose, smpl_shape, smpl_mesh = process_human_model_output(data['smpl_param'], data['cam_param'], do_flip, img_shape, img2bb_trans, rot)
            cam_trans = np.array(data['smpl_param']['trans'], dtype=np.float32)
            cam_param = np.array([cfg.focal[0], cfg.focal[1], cfg.princpt[0], cfg.princpt[1]])

            # gender
            if data['gender'] == 'male': gender = 1
            elif data['gender'] == 'female': gender = 2
            else: gender = 0

            if cfg.calculate_bcc:
                # load parse (cloth segmentation)
                parse = cv2.imread(data['parse_path'])
                parse_list = []
                for cloth_type in ('fg',) + cfg.cloth_types:
                    # get cloth indexs
                    if cloth_type == 'fg':
                        idxs = np.unique(parse).tolist()
                        idxs.pop(idxs.index(0))
                        if len(idxs) == 0:
                            parse_fg = np.zeros((cfg.output_parse_shape[0], cfg.output_parse_shape[1])) > 0
                            continue
                    else:
                        idxs = self.parse_set[cloth_type]
                    
                    # get masking corresponding to a cloth
                    mask = [parse == i for i in idxs]
                    mask = (sum(mask) > 0).astype(np.float32)
                    _, _, _, lip2img_trans = generate_patch_image(mask, data['orig_bbox'], 1.0, 0.0, False, mask.shape)
                    mask = cv2.warpAffine(mask, lip2img_trans, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)
                    mask = cv2.warpAffine(mask, img2bb_trans, (cfg.input_img_shape[1], cfg.input_img_shape[0]), flags=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (cfg.output_parse_shape[1], cfg.output_parse_shape[0]))
                    
                    if cloth_type == 'fg': parse_fg = mask[:,:,0] > 0
                    else: parse_list.append(mask)

                parse = np.stack(parse_list)[:,:,:,0] # parse: (cloths, height, width)
                is_bkg = (np.prod(parse == 0, 0) == 1)
                parse = np.argmax(parse, 0) + 1 # add 1 for bkg class
                parse[is_bkg] = 0
                parse[valid_mask == 0 ] = -1
                parse_valid = valid_mask

                # load densepose
                dp_xy, dp_uv, dp_I, dp_vertex = data['dp']['xy'], data['dp']['uv'], data['dp']['I'], data['dp']['smpl_v_idx']
                dp_xy = np.concatenate((dp_xy, np.ones_like(dp_xy[:,:1])),1)
                dp_xy = np.dot(img2bb_trans, dp_xy.transpose(1,0)).transpose(1,0)
                dp_I = dp_I - 1 # dp_I is started wtih 1. make it zero-based index.

                cur_point_num = len(dp_xy)
                if cur_point_num > cfg.dp_point_num:
                    idxs = np.random.choice(np.arange(cur_point_num), size=cfg.dp_point_num)
                    cur_point_num = cfg.dp_point_num
                    dp_xy = dp_xy[idxs]; dp_uv = dp_uv[idxs]; dp_I = dp_I[idxs]; dp_vertex = dp_vertex[idxs]
                
                # match densepose & parse
                _dp_xy = dp_xy.copy()
                _dp_xy[:,0] = _dp_xy[:,0] / cfg.input_img_shape[1] * cfg.output_parse_shape[1]
                _dp_xy[:,1] = _dp_xy[:,1] / cfg.input_img_shape[0] * cfg.output_parse_shape[0]

                parse_onehot = np.zeros((len(cfg.cloth_types)+1, cfg.output_parse_shape[0], cfg.output_parse_shape[1]))
                for i in range(len(cfg.cloth_types)+1):
                    parse_onehot[i][parse == i] = 1.0

                dp_cloth_idx = np.ones((_dp_xy.shape[0]), np.int16) * -1
                dp_cloth_idx[bilinear_interpolate(parse_fg[None,:,:], _dp_xy[:,0], _dp_xy[:,1])[0] > 0.5] = 0
                for i in range(len(cfg.cloth_types)):
                    dp_cloth_idx[np.argmax(bilinear_interpolate(parse_onehot, _dp_xy[:,0], _dp_xy[:,1]), 0) == (i+1)] = i+1

                smpl_cloth_idx = np.ones((smpl.vertex_num), dtype=np.int16) * -1
                smpl_cloth_idx[dp_vertex] = dp_cloth_idx
                smpl_cloth_valid = (smpl_cloth_idx != -1).astype(np.float32)
                smpl_patch_idx = np.ones((smpl.vertex_num), dtype=np.int16) * -1
                smpl_patch_idx[dp_vertex] = dp_I[dp_I != -1]

                # For BCC calculation, T-posed mesh is used.
                smpl_pose = np.zeros((smpl.joint_num*3,), dtype=np.float32)
                smpl_shape = np.zeros((smpl.shape_param_dim,), dtype=np.float32)
                cam_trans = np.zeros((3,), dtype=np.float32)
            else:
                smpl_cloth_idx = np.ones((smpl.vertex_num), dtype=np.int16) * -1
                smpl_patch_idx = np.ones((smpl.vertex_num), dtype=np.int16) * -1

            inputs = {'img': img}
            targets = {'gender': gender, 'smpl_cloth_idx': smpl_cloth_idx, 'smpl_patch_idx': smpl_patch_idx}
            meta_info = {'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape, 'cam_trans': cam_trans, 'cam_param': cam_param}
            return inputs, targets, meta_info


    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)

        eval_result = {'bcc_upper':[], 'bcc_lower':[], 'bcc_non_cloth':[]}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            ann_id = annot['ann_id']
            out = outs[n]
            
            if cfg.calculate_bcc:
                for cloth_type in cfg.cloth_types:
                    if out[cloth_type + '_mesh'] is None: 
                        out[cloth_type + '_mesh'] = np.array([])
                    else:
                        out[cloth_type + '_mesh'] = np.array(out[cloth_type + '_mesh'].vertices)
                
                total_pred_cloth_verts = np.zeros((len(out['smpl_mesh']),))
                gt_smpl_cloth_idx = out['smpl_cloth_idx_target']

                for cloth_type in self.eval_types:
                    if cloth_type == 'upper_body':
                        gt_cloth_verts = ((gt_smpl_cloth_idx == 1) |(gt_smpl_cloth_idx == 2)) # uppercloth, coat
                        cloth_idx = 1
                    elif cloth_type == 'lower_body':
                        gt_cloth_verts = ((gt_smpl_cloth_idx == 3 )| (gt_smpl_cloth_idx == 4)) # pants, skirts
                        cloth_idx = 2
                    elif cloth_type == 'non_cloth':
                        gt_cloth_verts = (gt_smpl_cloth_idx == 0) # non-cloth
                    
                    gt_idxs = np.where(gt_cloth_verts)[0]    

                    if cloth_type == 'upper_body':
                        cloth_verts = torch.cat([torch.from_numpy(out['uppercloth_mesh']).cuda(), torch.from_numpy(out['coat_mesh']).cuda()])
                        cloth_verts = cloth_verts[::8].float() # subsampling for memory efficiency

                    elif cloth_type == 'lower_body':
                        cloth_verts = torch.cat([torch.from_numpy(out['pants_mesh']).cuda(), torch.from_numpy(out['skirts_mesh']).cuda()])
                        cloth_verts = cloth_verts[::8].float() # subsampling for memory efficiency

                    smpl_verts = torch.tensor(out['smpl_mesh']).cuda()

                    if cloth_type in ['upper_body', 'lower_body']:
                        if len(cloth_verts) > 0:
                            dists = torch.sqrt(torch.sum((smpl_verts[None,:,:] - cloth_verts[:,None,:])**2,2))
                            dists = dists.min(0).values

                            pred_verts = (dists < self.bcc_dist_threshold).cpu().numpy()
                            total_pred_cloth_verts[pred_verts] = cloth_idx
                            correct_verts = (pred_verts[gt_idxs] == gt_cloth_verts[gt_idxs])
                        else:
                            correct_verts = np.zeros_like(gt_cloth_verts[gt_idxs], dtype=bool)
                    elif cloth_type == 'non_cloth':
                        pred_verts = (total_pred_cloth_verts == 0)
                        correct_verts = (pred_verts[gt_idxs] == gt_cloth_verts[gt_idxs])

                    if len(gt_idxs) == 0: continue    

                    if cloth_type == 'upper_body':
                        eval_result[f'bcc_upper'].append(correct_verts.sum()/len(gt_idxs))
                    elif cloth_type == 'lower_body':
                        eval_result[f'bcc_lower'].append(correct_verts.sum()/len(gt_idxs))
                    elif cloth_type == 'non_cloth':
                        eval_result[f'bcc_non_cloth'].append(correct_verts.sum()/len(gt_idxs))

        return eval_result

    def print_eval_result(self, eval_result):
        bcc_upper = np.mean(eval_result['bcc_upper'])
        bcc_lower = np.mean(eval_result['bcc_lower'])
        bcc_non_cloth = np.mean(eval_result['bcc_non_cloth'])
        bcc_average = (bcc_upper + bcc_lower + bcc_non_cloth) / 3

        print(">> BCC (upper body) : %.3f"%bcc_upper)
        print(">> BCC (lower body) : %.3f"%bcc_lower)
        print(">> BCC (non-cloth) : %.3f"%bcc_non_cloth)
        print(">> BCC (average) : %.3f"%bcc_average)

