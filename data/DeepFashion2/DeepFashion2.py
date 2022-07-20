import os
import os.path as osp
import numpy as np
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from utils.human_models import smpl
from utils.preprocessing import load_img, process_bbox, augmentation, generate_patch_image, bilinear_interpolate
from utils.vis import save_obj, vis_parse
from config import cfg

class DeepFashion2(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split

        if data_split != 'train':
            assert 0, "Invalid train mode."

        self.img_path = osp.join('..', 'data', 'DeepFashion2', 'data')
        self.annot_path = osp.join('..', 'data', 'DeepFashion2', 'data')
        self.parse_path = osp.join('..', 'data', 'preprocessed_data', 'parse', 'DeepFashion2')
        self.preprocessed_path = osp.join('..', 'data', 'preprocessed_data')
        self.dp_path = osp.join(self.preprocessed_path, 'densepose', 'DeepFashion2')

        # lip parse set
        self.parse_set = {'uppercloth': (5,), 'coat': (7,), 'pants': (9,), 'skirts': (12,), 'hair': (2,), 'shoes': (18,19)}
        self.sampling_stride = 4 # subsampling for training

        self.datalist = self.load_data()
        print("Load data: ", len(self.datalist))
        
    def load_data(self):
        self.img_path = osp.join(self.img_path , 'train', 'image')
        self.dp_path = osp.join(self.dp_path, 'train')
        db = COCO(osp.join(self.annot_path, 'DeepFashion2_train.json'))
        with open(osp.join(self.preprocessed_path, 'gender', 'DeepFashion2_train_gender.json')) as f:
            genders = json.load(f)
        with open(osp.join(self.parse_path, 'train_parsing_annotation.json')) as f:
            parsing_paths = json.load(f)

        datalist = []
        i = 0
        for aid in db.anns.keys():
            i += 1
            if i % self.sampling_stride != 0:
                continue

            ann = db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]
            img_path = osp.join(self.img_path, img['file_name'])
            
            # bbox
            bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
            if bbox is None: continue
            
            # parse
            if parsing_paths is not None:
                if str(aid) in parsing_paths:
                    parse_path = osp.join(self.parse_path, parsing_paths[str(aid)])
                else:
                    continue
            else:
                parse_path = None

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
            else:
                dp_data = None

            # gender
            if str(aid) in genders:
                gender = genders[str(aid)]
            else:
                continue
            
            data_dict = {
                'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 
                'bbox': bbox, 'orig_bbox': ann['bbox'], 'gender': gender, 
                'parse_path': parse_path, 'dp': dp_data
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

        parse = np.stack(parse_list)[:,:,:,0] # remove the last dimension (which has 3 channels)
        is_bkg = (np.prod(parse == 0, 0) == 1)
        parse = np.argmax(parse, 0) + 1 # add 1 to make the bkg class
        parse[is_bkg] = 0
        parse[valid_mask == 0] = -1
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