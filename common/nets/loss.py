import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.human_models import smpl
from utils.vis import save_obj
from config import cfg

class ClothClsLoss(nn.Module):
    def __init__(self):
        super(ClothClsLoss, self).__init__()
        self.dp_parts = {
            'head': [22,23],
            'upperbody': [0,1,14,15,16,17,18,19,20,21],
            'lowerbody': [6,7,8,9,10,11,12,13],
            'foot': [4,5]
        }
        self.part_clothes={
            'head': ['hair'],
            'upperbody': ['uppercloth', 'coat'],
            'lowerbody': ['pants', 'skirts'],
            'foot': ['shoes']
            }
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, out, patch_idx, cloth_idx):
        # valid only on visible
        valid = torch.zeros_like(out).cuda()
        index_gt = torch.zeros_like(out).cuda()

        for part in self.dp_parts.keys():
            valid_one_part = torch.zeros((out.shape[0],)).cuda()

            for part_idx in self.dp_parts[part]:
                valid_one_part += (patch_idx == part_idx).any(1)

            for cloth in self.part_clothes[part]:
                if cloth in cfg.cloth_types:
                    valid[valid_one_part>0, cfg.cloth_types.index(cloth)] = 1

        for idx in range(len(cfg.cloth_types)):
            index_gt[:, idx] += (cloth_idx==idx+1).any(1)
        
        loss = self.bce_loss(out, index_gt)
        loss = loss[valid>0]
        return loss.mean()


class GenderClsLoss(nn.Module):
    def __init__(self):
        super(GenderClsLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, out, gt):
        valid = (gt != 0) # if neutral gender, set valid = 0
        gt = F.one_hot((gt.long()), num_classes=3)[:,1:].float()
        
        loss = self.bce_loss(out, gt)
        loss = loss[valid]
        return loss.mean()


class SdfDPLoss(nn.Module):
    def __init__(self):
        super(SdfDPLoss, self).__init__()

    def forward(self, sdf, cloth_meshes_unposed, smpl_cloth_idx, smpl_cloth_valid, cloth_idx, sdf_thresh, dist_thresh, v_template):
        batch_size = sdf.shape[0]
        cloth_type = cfg.cloth_types[cloth_idx[0]-1]

        loss_list = []
        for bid in range(batch_size):
            smpl_mask = smpl_cloth_valid[bid] > 0
            smpl_verts = v_template[bid][smpl_mask[:,None].repeat(1,3)].view(-1,3)
            cloth_verts = cloth_meshes_unposed[bid]

            if smpl_verts.shape[0] > 0:
                dists = torch.sqrt(torch.sum((smpl_verts[None,:,:] - cloth_verts[:,None,:])**2,2))
            else:
                loss_list.append(torch.zeros((1)).mean().float().cuda())
                continue

            # remove too closest query points
            dists[dists<cfg.min_dist_thresh[cloth_type]] = 9999
            dists, query_point_idx = torch.min(dists,1)
            target_cloth_idx = smpl_cloth_idx[bid][smpl_mask]
            target_cloth_idx = target_cloth_idx[query_point_idx]
            
            # calculate loss
            loss_pos = torch.abs(sdf[bid,:]) * (sum([target_cloth_idx == idx for idx in cloth_idx]) > 0) * (dists < dist_thresh)
            loss_neg = torch.abs(sdf[bid,:] - sdf_thresh) * (sum([target_cloth_idx == idx for idx in cloth_idx]) == 0) * (dists < dist_thresh)

            cloth_exist = (sum([target_cloth_idx == idx for idx in cloth_idx]) > 0).sum() > 0
            loss = (loss_pos + loss_neg).mean() * cloth_exist
            loss_list.append(loss)
        
        loss = torch.stack(loss_list)
        return loss

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='none')

    def forward(self, param, valid):
        zeros = torch.zeros_like(param).cuda()
        loss = self.l2_loss(param, zeros) * valid[:,None]
        return loss.mean()

class SdfParseLoss(nn.Module):
    def __init__(self):
        super(SdfParseLoss, self).__init__()

    def forward(self, sdf, cloth_meshes, parse_gt, sdf_thresh, cloth_meshes_unposed, parse_valid, dist_thresh, v_template):
        batch_size = sdf.shape[0]
        inf = 9999
        
        # mask invalid xy coordinatets
        x, y = cloth_meshes[:,:,0].long(), cloth_meshes[:,:,1].long()
        idx = y * cfg.input_img_shape[1] + x
        is_valid = (x >= 0) * (x < cfg.input_img_shape[1]) * (y >= 0) * (y < cfg.input_img_shape[0])
        idx[is_valid == 0] = 0

        # minimum sdf
        min_sdf = sdf * is_valid.float() + inf * (1 - is_valid.float())
        parse_out_min = torch.ones((batch_size, cfg.input_img_shape[0] * cfg.input_img_shape[1])).float().cuda() * inf
        
        # maximum sdf
        max_sdf = sdf * is_valid.float() - inf * (1 - is_valid.float())
        parse_out_max = torch.ones((batch_size, cfg.input_img_shape[0] * cfg.input_img_shape[1])).float().cuda() * -inf

        try:
            parse_out_min, _ = scatter_min(min_sdf, idx, 1, parse_out_min)
            parse_out_max, _ = scatter_max(max_sdf, idx, 1, parse_out_max)
        except:
            # some GPUs have trouble in torch_scatter, compute in CPU
            idx = idx.cpu()
            min_sdf, max_sdf = min_sdf.cpu(), max_sdf.cpu()
            parse_out_min, parse_out_max = parse_out_min.cpu(), parse_out_max.cpu()
            parse_out_min, _ = scatter_min(min_sdf, idx, 1, parse_out_min)
            parse_out_max, _ = scatter_max(max_sdf, idx, 1, parse_out_max)
            parse_out_min, parse_out_max = parse_out_min.cuda(), parse_out_max.cuda()

        parse_out_min = parse_out_min.view(batch_size, cfg.input_img_shape[0], cfg.input_img_shape[1])
        parse_out_min[parse_out_min == inf] = 0

        parse_out_max = parse_out_max.view(batch_size, cfg.input_img_shape[0], cfg.input_img_shape[1])
        parse_out_max[parse_out_max == -inf] = sdf_thresh

        loss_pos = torch.abs(parse_out_min) * (parse_gt == 1) * parse_valid
        loss_neg = torch.abs(parse_out_max - sdf_thresh) * (parse_gt == 0) * parse_valid
        loss = loss_pos.mean((1,2)) + loss_neg.mean((1,2))

        cloth_exist = (parse_gt == 1).sum((1,2)) > 0
        loss = loss * cloth_exist 
        return loss