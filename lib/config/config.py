import open3d as o3d
from . import yacs
from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
import pprint

cfg = CN()

cfg.parent_cfg = 'configs/default.yaml'

# experiment name
cfg.exp_name = 'hello'

# network
cfg.point_feature = 9
cfg.distributed = False
cfg.num_latent_code = -1
cfg.fh_layers = 8

# data
cfg.data = CN()
cfg.data.hand_bound_reduce = 0.3
cfg.data.body_hand_bound_overlap = 0.1
cfg.data.face_bound_reduce = 0.1
cfg.data.body_face_bound_overlap = 0.1
cfg.data.face_mesh_extract_delta = 0.04

cfg.eval_mesh = CN()
cfg.eval_mesh.face_bound_add = 0.03
cfg.eval_mesh.hand_bound_add = 0.1


cfg.human = 313
cfg.training_view = [0, 6, 12, 18]
# hand
cfg.handl_training_view = [0, 6, 12, 18]
cfg.handr_training_view = [0, 6, 12, 18]
cfg.test_view = []
# hand
cfg.handl_test_view = []
cfg.handr_test_view = []
cfg.begin_ith_frame = 0  # the first smpl
cfg.num_train_frame = 1  # number of smpls
cfg.fix_num_train_frame = 1  # number of smpls
cfg.num_eval_frame = -1  # number of frames to render
cfg.ith_smpl = 0  # the i-th smpl
cfg.frame_interval = 1
cfg.smpl = 'smpl_4views_5e-4'
cfg.vertices = 'vertices'
cfg.params = 'params_4views_5e-4'
cfg.mask_bkgd = True
cfg.sample_smpl = False
cfg.sample_grid = False
cfg.sample_fg_ratio = 0.7
cfg.add_pointcloud = False
# face
cfg.face_begin_ith_frame = 0  # the first smpl
cfg.face_num_train_frame = 1  # number of smpls
cfg.fix_face_num_train_frame = 1  # number of smpls
cfg.face_frame_interval = 1
# hand
cfg.handl_begin_ith_frame = 0  # the first smpl
cfg.handl_num_train_frame = 1  # number of smpls
cfg.fix_handl_num_train_frame = 1  # number of smpls
cfg.handl_frame_interval = 1
cfg.handr_begin_ith_frame = 0  # the first smpl
cfg.handr_num_train_frame = 1  # number of smpls
cfg.fix_handr_num_train_frame = 1  # number of smpls
cfg.handr_frame_interval = 1
cfg.hand_repeat = 1
cfg.final_hand = False
cfg.hand_type = 'handl'

# process
cfg.process = CN()
cfg.process.smooth_rgb = False

# debug
cfg.debug = CN()
cfg.debug.use_mean_view_dir = False
cfg.debug.all_video_train = False
cfg.debug.single_part = False
cfg.debug.train_mesh = False


cfg.body = True
cfg.face = False
cfg.hand = False

cfg.big_box = False
cfg.box_padding = 0.05
cfg.face_box_padding = 0.02

cfg.rot_ratio = 0.
cfg.rot_range = np.pi / 32

# mesh
cfg.mesh_th = 50  # threshold of alpha
cfg.vis_union = False
# task
cfg.task = 'nerf4d'
cfg.type = ''
# gpus
cfg.gpus = list(range(8))
# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 100
cfg.save_latest_ep = 5
cfg.eval_ep = 100

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = ''
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0.

cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})

cfg.train.batch_size = 4

cfg.train.acti_func = 'relu'

cfg.train.use_vgg = False
cfg.train.vgg_pretrained = ''
cfg.train.vgg_layer_name = [0,0,0,0,0]

cfg.train.use_ssim = False
cfg.train.use_d = False
cfg.train_mask_mlp = False

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.test.frame_sampler_interval = 30

# debug
cfg.debug = CN()
cfg.debug.datasetVerticeProjection = False
cfg.debug.networkOptimizePose = True
cfg.debug.networkDeform = True
cfg.debug.networkBlendWeightOnMesh = True


# trained model
cfg.trained_model_dir = 'data/trained_model'

# recorder
cfg.record_dir = 'data/record'
cfg.log_interval = 20
cfg.record_interval = 20

# result
cfg.result_dir = 'data/result'

# training
cfg.training_mode = 'default'
cfg.train_with_coord = False
cfg.train_init_sdf = False
cfg.train_init_bw = False
cfg.train_union_sdf = False
cfg.tpose_viewdir = True
cfg.color_with_viewdir = True
cfg.color_with_feature = False
cfg.forward_rendering = False
cfg.smpl_rasterization = False
cfg.latent_optim = False
cfg.view_dirs_statics = False
cfg.face_vis_statics = False
cfg.mask_rendering = False
cfg.has_forward_resd = True
cfg.train_forward_resd = False
cfg.train_with_normal = False
cfg.tpose_geometry = True
cfg.erode_edge = True
cfg.num_trained_mask = 3
cfg.init_handl = 'no'
cfg.init_handr = 'no'
cfg.init_face = 'no'

cfg.trick_sample = False
cfg.train_bgfg = False

# loss weights
cfg.bw_loss_weight = 1
cfg.pose_loss_weight = 0
cfg.normal_loss_weight = 0.1
cfg.train_with_overlap = False
cfg.train_with_smooth_viewdir = False
cfg.grad_loss_weigth = 0.1
cfg.smooth_sdf_weight = [[0, 10000, 20000, 30000], [0, 0.1, 0.5, 1]]
# cfg.smooth_sdf_weight = 0.1
# fix part network params
cfg.fix_body = False
cfg.fix_face = False
cfg.fix_handl = False
cfg.fix_handr = False

# evaluation
cfg.eval = False
cfg.skip_eval = False
cfg.test_novel_pose = False
cfg.novel_pose_ni = 100
cfg.vis_novel_pose = False
cfg.vis_novel_view = False
cfg.vis_tpose_mesh = False
cfg.vis_face_mesh = False
cfg.vis_hand_mesh = False
cfg.vis_posed_mesh = False
cfg.vis_train_view = False
# cfg.train_full = False
cfg.vis_train_begin_i = 0
cfg.vis_train_ni = 0
cfg.vis_train_interval = 1
cfg.view_dir_path = ''

cfg.vis_mesh_multi = True
cfg.bg_color = 0.

cfg.fix_random = False

cfg.vis = 'mesh'

# data
cfg.body_sample_ratio = 0.5
cfg.face_sample_ratio = 0.
cfg.interhand_sample_ratio = 0.
cfg.new_hand = False
cfg.motion_data = 'TCD'
#eval
cfg.evaluation = CN()
cfg.evaluation.neuralbody = False
cfg.evaluation.aninerf = False

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    if cfg.num_latent_code < 0:
        cfg.num_latent_code = cfg.num_train_frame

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    if cfg.forward_rendering:
        cfg.result_dir = cfg.result_dir + '_fw'
    if cfg.evaluation.neuralbody:
        cfg.result_dir = cfg.result_dir + '_nb'
    if cfg.evaluation.aninerf:
        cfg.result_dir = cfg.result_dir + '_aninerf'

    cfg.local_rank = args.local_rank
    cfg.distributed = cfg.distributed or args.launcher not in ['none']
    cfg.type = args.type


def make_cfg(args):
    with open(args.cfg_file, 'r') as f:
        current_cfg = yacs.load_cfg(f)

    # if 'parent_cfg' in current_cfg.keys():
    #     with open(current_cfg.parent_cfg, 'r') as f:
    #         parent_cfg = yacs.load_cfg(f)
    #     cfg.merge_from_other_cfg(parent_cfg)

    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = yacs.load_cfg(f)
        if 'parent_cfg' in parent_cfg.keys():
            with open(parent_cfg.parent_cfg, 'r') as f:
                parent2_cfg = yacs.load_cfg(f)
            cfg.merge_from_other_cfg(parent2_cfg)
        cfg.merge_from_other_cfg(parent_cfg)

    cfg.merge_from_other_cfg(current_cfg)
    cfg.merge_from_list(args.opts)

    if cfg.train_init_sdf:
        cfg.merge_from_other_cfg(cfg.train_init_sdf_cfg)

    if cfg.train_init_bw:
        cfg.merge_from_other_cfg(cfg.train_init_bw_cfg)

    if cfg.train_forward_resd:
        cfg.has_forward_resd = True
        cfg.merge_from_other_cfg(cfg.train_forward_resd_cfg)

    if cfg.color_with_feature:
        cfg.merge_from_other_cfg(cfg.color_feature_cfg)

    if cfg.forward_rendering:
        cfg.has_forward_resd = True
        if cfg.debug.single_part:
            cfg.merge_from_other_cfg(cfg.forward_rendering_single_cfg)
        else:
            cfg.merge_from_other_cfg(cfg.forward_rendering_cfg)

    if cfg.vis_novel_pose:
        cfg.merge_from_other_cfg(cfg.novel_pose_cfg)

    if cfg.vis_novel_view:
        cfg.merge_from_other_cfg(cfg.novel_view_cfg)

    if cfg.vis_tpose_mesh or cfg.vis_posed_mesh:
        if cfg.debug.single_part:
            cfg.merge_from_other_cfg(cfg.mesh_single_part_cfg)
        else:
            cfg.merge_from_other_cfg(cfg.mesh_cfg)
        if cfg.debug.train_mesh:
            cfg.merge_from_other_cfg(cfg.mesh_train_cfg)


    if cfg.vis_train_view:
        cfg.merge_from_other_cfg(cfg.train_view_cfg)

    if cfg.train_union_sdf:
        cfg.merge_from_other_cfg(cfg.train_union_sdf_cfg)

    if cfg.mask_rendering:
        cfg.merge_from_other_cfg(cfg.mask_rendering_cfg)

    if cfg.smpl_rasterization:
        cfg.merge_from_other_cfg(cfg.smpl_rasterization_cfg)

    if cfg.latent_optim:
        cfg.merge_from_other_cfg(cfg.latent_optim_cfg)

    if cfg.view_dirs_statics:
        cfg.merge_from_other_cfg(cfg.view_dirs_statics_cfg)

    if cfg.face_vis_statics:
        cfg.merge_from_other_cfg(cfg.face_vis_statics_cfg)

    # if cfg.train_full:
    #     cfg.merge_from_other_cfg(cfg.train_full_cfg)


    cfg.merge_from_list(args.opts)

    parse_cfg(cfg, args)
    # pprint.pprint(cfg)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
