# face
python train_net.py --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_face_bound0.1 init_sdf sq_smplh_multi_2 train.epoch 400 fix_body True fix_face False fix_handl True fix_handr True handl_num_train_frame 0 handr_num_train_frame 0 num_train_frame 0 ratio 0.5 gpus 1,
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_face_bound0.1 resume True vis_tpose_mesh True vis_face_mesh True num_train_frame 1 train.num_workers 0 part_type 1 gpus 1, debug.single_part True debug.transform_face False

# handl
python train_net.py --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_handl init_sdf sq_smplh_multi_2 train.epoch 400 fix_body True fix_face True fix_handl False fix_handr True face_num_train_frame 0 handr_num_train_frame 0 num_train_frame 0 ratio 0.5 gpus 2,
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_handl resume True vis_tpose_mesh True num_train_frame 1 train.num_workers 0 part_type 2 gpus 1, debug.single_part True debug.transform_face False vis_hand_mesh True hand_type handl

# handr
python train_net.py --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_handr init_sdf sq_smplh_multi_2 train.epoch 400 fix_body True fix_face True fix_handl True fix_handr False handl_num_train_frame 0 face_num_train_frame 0 num_train_frame 0 ratio 0.5 gpus 3,
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_handr resume True vis_tpose_mesh True num_train_frame 1 train.num_workers 0 part_type 3 gpus 1,  debug.single_part True debug.transform_face False vis_hand_mesh True hand_type handr

# body
python train_net.py --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_body init_sdf sq_smplh_multi_2 train.epoch 400 fix_body False fix_face True fix_handl True fix_handr True handl_num_train_frame 0 handr_num_train_frame 0 face_num_train_frame 0 ratio 0.5 gpus 3,
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_body resume True vis_tpose_mesh True num_train_frame 1 train.num_workers 0 part_type 0 gpus 1, debug.single_part True debug.transform_face False
#python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_old_tshirt_body resume True vis_tpose_mesh True num_train_frame 1 train.num_workers 0 part_type 0 gpus 1, debug.single_part True debug.transform_face False
# register body-face and face
python tools/finetune_face_pose.py --path ./data/animation/release_tshirt_face_bound0.1 --bodypath ./data/animation/release_old_tshirt_body/ --debug
# full
python train_net.py --cfg_file configs/anisdf_zju_djt415_tshirt.yaml  exp_name release_tshirt_full init_sdf sq_smplh_multi_2 train.epoch 400 fix_body False fix_face True fix_handl True fix_handr True handl_num_train_frame 0 handr_num_train_frame 0 face_num_train_frame 0  ratio 0.5 init_face release_tshirt_face_bound0.1 init_handl release_tshirt_handl init_handr release_tshirt_handr trick_sample True gpus 1,
# remove the debug.single_part flag
# obtain the mesh
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_full resume True vis_tpose_mesh True num_train_frame 1 train.num_workers 0 part_type 0 gpus 1, init_face release_tshirt_face_bound0.1
# obtain the ray transformation
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_full resume True view_dirs_statics True gpus 3, ratio 0.5 init_face release_tshirt_face_bound0.1
# forward rendering with novel pose; you need to provide the novel human poses. We download the mocap data from AMASS.
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_full resume True  forward_rendering True  handl_num_train_frame 0 handr_num_train_frame 0 face_num_train_frame 0  forward_rendering_cfg.z_interval 0.005 ratio 1 bg_color 0  N_samples 3  debug.use_mean_view_dir True vis_novel_pose True init_face release_tshirt_face_bound0.1 gpus 2,
# appearance composition, latent optimization;
# you can select to optimize those parts, whose lightings are quite different.
# optim right hand
python train_net.py --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_full_optim_hr init_sdf release_tshirt_full resume True gpus '2,' latent_optim True part_type 3  init_face release_tshirt_face_bound0.1 debug.use_mean_view_dir True view_dir_path data/view_dirs/release_tshirt_full/view_dirs.npy gpus 2,
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_full_optim_hr resume True  forward_rendering True  handl_num_train_frame 0 handr_num_train_frame 0 face_num_train_frame 0  forward_rendering_cfg.z_interval 0.005 ratio 1 bg_color 0  N_samples 3  debug.use_mean_view_dir True vis_novel_pose True init_face release_tshirt_face_bound0.1 view_dir_path data/view_dirs/release_tshirt_full/view_dirs.npy gpus 2,
# optim left hand
python train_net.py --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_full_optim_hrhl init_sdf release_tshirt_full_optim_hr resume True gpus '2,' latent_optim True part_type 2  init_face release_tshirt_face_bound0.1 debug.use_mean_view_dir True view_dir_path data/view_dirs/release_tshirt_full/view_dirs.npy gpus 2,
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_full_optim_hrhl resume True  forward_rendering True  handl_num_train_frame 0 handr_num_train_frame 0 face_num_train_frame 0  forward_rendering_cfg.z_interval 0.005 ratio 1 bg_color 0  N_samples 3  debug.use_mean_view_dir True vis_novel_pose True init_face release_tshirt_face_bound0.1 view_dir_path data/view_dirs/release_tshirt_full/view_dirs.npy gpus 2,
# optim face
python train_net.py --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_full_optim_hrhlf init_sdf release_tshirt_full_optim_hrhl resume True gpus '2,' latent_optim True part_type 1  init_face release_tshirt_face_bound0.1 debug.use_mean_view_dir True view_dir_path data/view_dirs/release_tshirt_full/view_dirs.npy gpus 2,
python run.py --type visualize --cfg_file configs/anisdf_zju_djt415_tshirt.yaml exp_name release_tshirt_full_optim_hrhlf resume True  forward_rendering True  handl_num_train_frame 0 handr_num_train_frame 0 face_num_train_frame 0  forward_rendering_cfg.z_interval 0.005 ratio 1 bg_color 0  N_samples 3  debug.use_mean_view_dir True vis_novel_pose True init_face release_tshirt_face_bound0.1 view_dir_path data/view_dirs/release_tshirt_full/view_dirs.npy gpus 2,


