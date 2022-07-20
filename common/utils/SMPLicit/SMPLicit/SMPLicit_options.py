import torch
import os
import numpy as np

# HUMAN PARSING LABELS:
# 1 -> Hat
# 2 -> Hair
# 3 -> Glove
# 4 -> Sunglasses,
# 5 -> Upper-Clothes,
# 6 -> Dress,
# 7 -> Coat,
# 8 -> Socks,
# 9 -> Pants,
# 10 -> Torso-Skin
# 11 -> Scarf
# 12 -> Skirt
# 13 -> Face
# 14 -> Left Arm
# 15 -> Right Arm
# 16 -> Left Leg
# 17 -> Right Leg
# 18 -> Left Shoe
# 19 -> Right Shoe

class Options():
    def __init__(self):
        # Upper body options:
        self.upperbody_loadepoch = 11
        self.upperbody_clusters = 'indexs_clusters_tshirt_smpl.npy'
        self.upperbody_num_clusters = 500
        self.upperbody_n_z_cut = 6
        self.upperbody_n_z_style = 12
        self.upperbody_resolution = 128
        self.upperbody_thresh_occupancy = -0.03
        self.coat_thresh_occupancy = -0.08

        # Pants options:
        self.pants_loadepoch = 60
        self.pants_clusters = 'clusters_lowerbody.npy'
        self.pants_num_clusters = 500
        self.pants_n_z_cut = 6
        self.pants_n_z_style = 12
        self.pants_resolution = 128
        self.pants_thresh_occupancy = -0.02

        # Skirts options:
        self.skirts_loadepoch = 40
        self.skirts_clusters = 'clusters_lowerbody.npy'
        self.skirts_num_clusters = 500
        self.skirts_n_z_cut = 6
        self.skirts_n_z_style = 12
        self.skirts_resolution = 128
        self.skirts_thresh_occupancy = -0.05

        # Hair options:
        self.hair_loadepoch = 20000
        self.hair_clusters = 'clusters_hairs.npy'
        self.hair_num_clusters = 500
        self.hair_n_z_cut = 6
        self.hair_n_z_style = 12
        self.hair_resolution = 128
        self.hair_thresh_occupancy = -2.0

        # Shoes options
        self.shoes_loadepoch = 20000
        self.shoes_clusters = 'clusters_shoes.npy'
        self.shoes_n_z_cut = 0
        self.shoes_n_z_style = 4
        self.shoes_resolution = 64
        self.shoes_thresh_occupancy = -0.36
        self.shoes_num_clusters = 100

        # General options:
        self.path_checkpoints = '../../../../data/base_data/smplicit/checkpoints/'
        self.path_cluster_files = '../../../../data/base_data/smplicit/clusters/'
        self.path_SMPL = '../../../../data/base_data/human_models/smpl'

        self.upperbody_b_min = [-0.8, -0.4, -0.3]
        self.upperbody_b_max = [0.8, 0.6, 0.3]
        self.pants_b_min = [-0.3, -1.2, -0.3]
        self.pants_b_max = [0.3, 0.0, 0.3]
        self.skirts_b_min = [-0.3, -1.2, -0.3]
        self.skirts_b_max = [0.3, 0.0, 0.3]
        self.hair_b_min = [-0.35, -0.42, -0.33]
        self.hair_b_max = [0.35, 0.68, 0.37]
        self.shoes_b_min = [-0.1, -1.4, -0.2]
        self.shoes_b_max = [0.25, -0.6, 0.3]

