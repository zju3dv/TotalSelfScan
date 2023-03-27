import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import glob
import numpy as np
import copy
import json
import os
import os.path as osp
import imageio
import cv2
from lib.config import cfg
from lib.utils.blend_utils import *
from tools.rasterizer_mesh import set_pytorch3d_intrinsic_matrix
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils.debugger import dbg
from . import tpose_bfh_multi_dataset


class Dataset(tpose_bfh_multi_dataset.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__(data_root, human, ann_file, split)
        novel_pose = np.load('data/mocap/dmpl_sample.npz')
        novel_pose = np.load('data/mocap/amass_sample.npz')
        action_path = 'data/mocap/ACCAD/ACCAD/Female1General_c3d/*.npz'
        action_path = 'data/mocap/TCDHands/ExperimentDatabase/*.npz'
        actions = glob.glob(action_path)
        novel_poses = []
        for action_i in actions:
            novel_pose_i = np.load(action_i)
            if 'poses' not in novel_pose_i.files:
                continue
            smplh_poses = np.concatenate([novel_pose_i['root_orient'],
                                          novel_pose_i['pose_body'],
                                          novel_pose_i['pose_hand']], axis=1)
            novel_poses.append(smplh_poses)
            # novel_poses.append(novel_pose_i['poses'])
        self.novel_pose = np.vstack(novel_poses)
        self.novel_pose[:, :3] = 0
        self.novel_pose = self.novel_pose[:]
        self.num_pose = self.novel_pose.shape[0]

        self.get_face_part_label()

    def get_face_part_label(self):
        tpose_dir = 'data/animation/{}/tpose_mesh.npy'.format(cfg.exp_name)
        tpose_mesh = np.load(tpose_dir, allow_pickle=True).item()
        self.tvertex = tpose_mesh['vertex']
        self.tface = tpose_mesh['triangle'].astype(np.int32)
        pts_of_handl = self.get_inside(self.tvertex, self.tbounds_handl)
        face_of_handl = pts_of_handl[self.tface]
        face_of_handl = face_of_handl.sum(axis=1) == 3
        pts_of_handr = self.get_inside(self.tvertex, self.tbounds_handr)
        face_of_handr = pts_of_handr[self.tface]
        face_of_handr = face_of_handr.sum(axis=1) == 3
        face_of_body = (~face_of_handl) * (~face_of_handr)
        face_label = np.zeros(self.tface.shape[0])
        face_label[face_of_handl] = 2
        face_label[face_of_handr] = 3
        self.face_label = face_label




    def prepare_input(self, i, index, part='body'):
        if part == 'body':
            # read xyz in the world coordinate system
            vertices_path = os.path.join(self.data_root, cfg.vertices,
                                         '{}.npy'.format(i))
            wxyz = np.load(vertices_path).astype(np.float32)

            # transform smpl from the world coordinate to the smpl coordinate
            params_path = os.path.join(self.data_root, cfg.params,
                                       '{}.npy'.format(i))
            params = np.load(params_path, allow_pickle=True).item()
            Rh = params['Rh'].astype(np.float32)
            Th = params['Th'].astype(np.float32)
            Th[:, 2] += 0.5
            # Rh += np.random.rand(Rh.shape[0], Rh.shape[1])/2
            # prepare sp input of param pose
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            pxyz = np.dot(wxyz - Th, R).astype(np.float32)

            # calculate the skeleton transformation
            poses = self.novel_pose[index:index+1].reshape(-1, 3)
            # poses = params['poses'].reshape(-1, 3)
            # poses += np.random.rand(poses.shape[0], poses.shape[1])/2
            joints = self.joints
            parents = self.parents
            A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)

            big_poses = np.zeros_like(poses).ravel()
            if cfg.tpose_geometry:
                angle = 30
                big_poses[5] = np.deg2rad(angle)
                big_poses[8] = np.deg2rad(-angle)
            else:
                big_poses = big_poses.reshape(-1, 3)
                big_poses[1] = np.array([0, 0, 7. / 180. * np.pi])
                big_poses[2] = np.array([0, 0, -7. / 180. * np.pi])
                big_poses[16] = np.array([0, 0, -55. / 180. * np.pi])
                big_poses[17] = np.array([0, 0, 55. / 180. * np.pi])
            big_poses = big_poses.reshape(-1, 3)
            big_A = if_nerf_dutils.get_rigid_transformation(
                big_poses, joints, parents)
            pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
            pbw = pbw.astype(np.float32)

            return wxyz, pxyz, A, big_A, pbw, Rh, Th

        elif part == 'handl':
            # read xyz in the world coordinate system
            vertices_path = os.path.join(self.handl_data_root, cfg.vertices,
                                         '{}.npy'.format(i))
            wxyz = np.load(vertices_path).astype(np.float32)

            # transform smpl from the world coordinate to the smpl coordinate
            params_path = os.path.join(self.handl_data_root, cfg.params,
                                       '{}.npy'.format(i))
            params = np.load(params_path, allow_pickle=True).item()
            Rh = params['Rh'].astype(np.float32)
            Th = params['Th'].astype(np.float32)

            # prepare sp input of param pose
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            pxyz = np.dot(wxyz - Th, R).astype(np.float32)

            # calculate the skeleton transformation
            poses = params['poses'].reshape(-1, 3)
            joints = self.handl_joints
            parents = self.handl_parents
            A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)

            big_poses = np.zeros_like(poses).ravel()
            if cfg.tpose_geometry:
                angle = 30
                big_poses[5] = np.deg2rad(angle)
                big_poses[8] = np.deg2rad(-angle)
            else:
                big_poses = big_poses.reshape(-1, 3)
                big_poses[1] = np.array([0, 0, 7. / 180. * np.pi])
                big_poses[2] = np.array([0, 0, -7. / 180. * np.pi])
                big_poses[16] = np.array([0, 0, -55. / 180. * np.pi])
                big_poses[17] = np.array([0, 0, 55. / 180. * np.pi])
            big_poses = big_poses.reshape(-1, 3)
            big_A = if_nerf_dutils.get_rigid_transformation(
                big_poses, joints, parents)
            pbw = np.load(os.path.join(self.handl_lbs_root, 'bweights/{}.npy'.format(i)))
            pbw = pbw.astype(np.float32)

            return wxyz, pxyz, A, big_A, pbw, Rh, Th
        elif part == 'handr':
            # read xyz in the world coordinate system
            vertices_path = os.path.join(self.handr_data_root, cfg.vertices,
                                         '{}.npy'.format(i))
            wxyz = np.load(vertices_path).astype(np.float32)

            # transform smpl from the world coordinate to the smpl coordinate
            params_path = os.path.join(self.handr_data_root, cfg.params,
                                       '{}.npy'.format(i))
            params = np.load(params_path, allow_pickle=True).item()
            Rh = params['Rh'].astype(np.float32)
            Th = params['Th'].astype(np.float32)

            # prepare sp input of param pose
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            pxyz = np.dot(wxyz - Th, R).astype(np.float32)

            # calculate the skeleton transformation
            poses = params['poses'].reshape(-1, 3)
            joints = self.handr_joints
            parents = self.handr_parents
            A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)

            big_poses = np.zeros_like(poses).ravel()
            if cfg.tpose_geometry:
                angle = 30
                big_poses[5] = np.deg2rad(angle)
                big_poses[8] = np.deg2rad(-angle)
            else:
                big_poses = big_poses.reshape(-1, 3)
                big_poses[1] = np.array([0, 0, 7. / 180. * np.pi])
                big_poses[2] = np.array([0, 0, -7. / 180. * np.pi])
                big_poses[16] = np.array([0, 0, -55. / 180. * np.pi])
                big_poses[17] = np.array([0, 0, 55. / 180. * np.pi])
            big_poses = big_poses.reshape(-1, 3)
            big_A = if_nerf_dutils.get_rigid_transformation(
                big_poses, joints, parents)
            pbw = np.load(os.path.join(self.handr_lbs_root, 'bweights/{}.npy'.format(i)))
            pbw = pbw.astype(np.float32)

            return wxyz, pxyz, A, big_A, pbw, Rh, Th


    def __getitem__(self, index):
        if cfg.final_hand:
            # increase the portion of hand image
            if self.total_label[index] <= 1:
                # if np.random.rand() <= 0.5:
                index = self.total_group[-1] + np.random.randint(len(self.handl_ims))
        if cfg.vis_train_view:
            index = cfg.vis_train_begin_i + index * cfg.vis_train_interval

        index = index * 3
        ori_index = index
        if True:
            # body part
            img_path_pose = os.path.join(self.data_root, self.ims[0])
            img_path_RhTh = os.path.join(self.data_root, self.ims[index])
            index = 100
            img = imageio.imread(img_path_pose).astype(np.float32) / 255.
            msk, orig_msk = self.get_mask(0)

            H, W = img.shape[:2]
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)

            cam_ind = self.cam_inds[0]
            K = np.array(self.cams['K'][cam_ind])
            D = np.array(self.cams['D'][cam_ind])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            orig_msk = cv2.undistort(orig_msk, K, D)

            R = np.array(self.cams['R'][cam_ind])
            T = np.array(self.cams['T'][cam_ind]) / 1000.

            # reduce the image resolution by ratio
            H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
            orig_H, orig_W = H, W
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            if cfg.mask_bkgd:
                img[msk == 0] = 1 * cfg.bg_color
            K[:2] = K[:2] * cfg.ratio

            if self.human in ['CoreView_313', 'CoreView_315']:
                i = int(os.path.basename(img_path).split('_')[4])
                frame_index = i - 1
            elif self.human in ['male-tzr-smplh']:
                i = os.path.basename(img_path_RhTh).split('.')[0]
                frame_index = ori_index
            else:
                i = int(os.path.basename(img_path)[:-4])
                frame_index = i

            # read v_shaped
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
            tpose = np.load(vertices_path).astype(np.float32)
            # import ipdb; ipdb.set_trace(context=11)
            # from lib.utils.debugger import dbg
            # dbg.showL3D([tpose])
            tbounds = if_nerf_dutils.get_bounds(tpose)
            tbw = np.load(os.path.join(self.lbs_root, 'bigpose_bw.npy'))
            tbw = tbw.astype(np.float32)

            wpts, ppts, A, big_A, pbw, Rh, Th = self.prepare_input(i, index)

            pbounds = if_nerf_dutils.get_bounds(ppts)
            wbounds = if_nerf_dutils.get_bounds(wpts)

            if False:
                # from IPython import embed;embed()
                wpts_c = (wpts @ R.T + T.T) @ K.T
                project_p = wpts_c[:, :2] / wpts_c[:, 2:]
                import matplotlib.pylab as plt;
                plt.figure();
                plt.imshow(img);
                plt.plot(project_p[:, 0], project_p[:, 1], 'r*');
                plt.show()
                import ipdb;
                ipdb.set_trace(context=11)

            pose = np.concatenate([R, T], axis=1)
            bound_mask = if_nerf_dutils.get_bound_2d_mask(wbounds, K, pose, H, W)
            img, msk, K, crop_bbox = if_nerf_dutils.crop_image_msk(
                img, msk, K, bound_mask)
            H, W = img.shape[:2]
            pytorch3d_K = set_pytorch3d_intrinsic_matrix(K, H, W)
            ray_o, ray_d, near, far, mask_at_box = if_nerf_dutils.get_rays_within_bounds_test(
                H, W, K, R, T, wbounds)

            if cfg.erode_edge:
                orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
            # occupancy = orig_msk[coord[:, 0], coord[:, 1]]
            rgb = img[mask_at_box == 1]
            img = img.transpose(2, 0, 1)
            msk = (msk == 1).astype(np.uint8)
            smpl_R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            # nerf
            ret = {
                'rgb': rgb,
                # 'occupancy': occupancy,
                'img': img,
                'msk': msk,
                'ray_o': ray_o,
                'ray_d': ray_d,
                'near': near,
                'far': far,
                'mask_at_box': mask_at_box,
                'R': smpl_R,
                'Th': Th,
                'cam_R': R,
                'cam_T': T,
                'pytorch3d_K': pytorch3d_K
            }

            # blend weight
            meta = {
                'A': A,
                'big_A': big_A,
                'pbw': pbw,
                'tbw': tbw,
                'pbounds': pbounds,
                'wbounds': wbounds,
                'tbounds': tbounds
            }
            ret.update(meta)

            latent_index = index // self.num_cams  # + (cfg.begin_ith_frame//cfg.frame_interval)
            if cfg.test_novel_pose:
                latent_index = cfg.num_train_frame - 1
            meta = {
                'H': H,
                'W': W,
                'orig_H': orig_H,
                'orig_W': orig_W,
                'crop_bbox': crop_bbox,
                'latent_index': latent_index,
                'frame_index': frame_index,
                'cam_ind': cam_ind,
                'view_index': cam_ind,
                'part_type': 0

            }
            ret.update(meta)
            if cfg.latent_optim:
                ret['part_type'] = cfg.part_type


        vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        ret.update({'tpose': tpose})
        ret.update({'tbounds_body': self.tbounds_body,
                    'tbounds_face': self.tbounds_face,
                    'tbounds_handl': self.tbounds_handl,
                    'tbounds_handr': self.tbounds_handr,
                    'face_label': self.face_label})
        ret.update({'total_group_fix': np.array(self.total_group_fix)})
        return ret

    def __len__(self):
        if cfg.vis_train_view:
            return cfg.vis_train_ni
        return len(self.ims)
