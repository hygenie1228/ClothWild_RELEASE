import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers
from utils.human_models import smpl
from config import cfg

class ClothNet(nn.Module):
    def __init__(self):
        super(ClothNet, self).__init__()
        input_feat_dim = 2048
        if 'uppercloth' in cfg.cloth_types:
            self.z_cut_uppercloth = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_uppercloth = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'coat' in cfg.cloth_types:
            self.z_cut_coat = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_coat = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'pants' in cfg.cloth_types:
            self.z_cut_pants = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_pants = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'skirts' in cfg.cloth_types:
            self.z_cut_skirts = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_skirts = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'hair' in cfg.cloth_types:
            self.z_cut_hair = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_hair = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'shoes' in cfg.cloth_types:
            self.z_style_shoes = make_linear_layers([input_feat_dim,4], relu_final=False)

        self.cloth_cls_layer = make_linear_layers([input_feat_dim, len(cfg.cloth_types)], relu_final=False)
        self.gender_cls_layer = make_linear_layers([input_feat_dim, 2], relu_final=False)
            

    def forward(self, img_feat):
        batch_size = img_feat.shape[0]
        img_feat = img_feat.mean((2,3))
        
        z_cuts, z_styles = [], []
        for cloth_type in cfg.cloth_types:
            if cloth_type == 'uppercloth':
                z_cuts.append(self.z_cut_uppercloth(img_feat))
                z_styles.append(self.z_style_uppercloth(img_feat))
            elif cloth_type == 'coat':
                z_cuts.append(self.z_cut_coat(img_feat))
                z_styles.append(self.z_style_coat(img_feat))
            elif cloth_type == 'pants':
                z_cuts.append(self.z_cut_pants(img_feat))
                z_styles.append(self.z_style_pants(img_feat))
            elif cloth_type == 'skirts':
                z_cuts.append(self.z_cut_skirts(img_feat))
                z_styles.append(self.z_style_skirts(img_feat))
            elif cloth_type == 'hair':
                z_cuts.append(self.z_cut_hair(img_feat))
                z_styles.append(self.z_style_hair(img_feat))
            elif cloth_type == 'shoes':
                z_cuts.append(torch.zeros((batch_size,0)).float().cuda())
                z_styles.append(self.z_style_shoes(img_feat))

        scores = self.cloth_cls_layer(img_feat)
        scores = torch.sigmoid(scores)

        genders = self.gender_cls_layer(img_feat)
        genders = F.softmax(genders, dim=-1)

        return genders, scores, z_cuts, z_styles
