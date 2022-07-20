import os
import os.path as osp
import sys
import numpy as np
import torch

class Config:
    ## dataset
    trainset_3d = []
    trainset_2d = ['MSCOCO', 'DeepFashion2']
    testset = ['MSCOCO', 'PW3D']

    ## model setting
    resnet_type = 50
    input_img_shape = (256, 192) 
    output_parse_shape = (256, 192)
    output_dp_shape = (64, 48)
    output_uv_shape = (64, 64)
    focal = (5000, 5000)                                    # virtual focal lengths
    princpt = (input_img_shape[1]/2, input_img_shape[0]/2)  # virtual principal point position
    dp_point_num = 196
    dp_patch_num = 24
    cloth_types = ('uppercloth', 'coat', 'pants', 'skirts', 'shoes')

    ## training config
    lr = 1e-4
    lr_dec_factor = 10
    lr_dec_epoch = [5]
    end_epoch = 8
    train_batch_size = 8
    sdf_thresh = {'uppercloth': 0.1, 'coat': 0.1, 'pants': 0.1, 'skirts': 0.1, 'shoes': 0.01}
    dist_thresh = {'uppercloth': 0.03, 'coat': 0.1, 'pants': 0.03, 'skirts': 0.03, 'shoes': 0.03}
    min_dist_thresh = {'uppercloth': 0.0, 'coat': 0.03, 'pants': 0.0, 'skirts': 0.0, 'shoes': 0.0}

    cls_weight = 0.01
    gender_weight = 0.01
    dp_weight = 1.0
    reg_weight = 0.1
    cloth_reg_weight = {'uppercloth': 1.0, 'coat': 1.0, 'pants': 1.0, 'skirts': 1.0, 'shoes': 0.1}

    ## testing config
    test_batch_size = 1
    cls_threshold = 0.25
    calculate_cd = False
    calculate_bcc = False
    cloth_colors = {'smpl_body':(190,190,190), 'uppercloth':(140,110,160), 'coat':(170,120,60), 'pants':(110,130,100), 'skirts':(90,110,140), 'shoes':(120,60,60)}

    num_thread = 8
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join(data_dir, 'base_data', 'human_models')
    smplicit_path = osp.join(root_dir, 'common', 'utils', 'SMPLicit', 'SMPLicit')

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

        if self.num_gpus != 1:
            assert 0, "Not support DataParallel."

cfg = Config()

np.random.seed(0)
torch.manual_seed(0)

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
for i in range(len(cfg.testset)):
    add_pypath(osp.join(cfg.data_dir, cfg.testset[i]))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)