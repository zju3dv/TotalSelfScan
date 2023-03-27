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
        if cfg.debug.transform_face:
            face_coord_transform_file = osp.join('data/animation', cfg.init_face, 'transform.npy')
            if osp.exists(face_coord_transform_file):
                face_coord_transform = np.load(face_coord_transform_file, allow_pickle=True).item()
                R_face2canonical = face_coord_transform['R_f'][0]
                T_face2canonical = face_coord_transform['T_f']
                self.R_face2canonical = R_face2canonical
                self.T_face2canonical = T_face2canonical

    def __getitem__(self, index):
        if cfg.final_hand:
            # increase the portion of hand image
            if self.total_label[index] <= 1:
                # if np.random.rand() <= 0.5:
                index = self.total_group[-1] + np.random.randint(len(self.handl_ims))
        if cfg.vis_train_view:
            index = cfg.vis_train_begin_i + index * cfg.vis_train_interval

        ori_index = index
        if self.human_part[self.total_label[index]] == 'body':
            # body part
            index = index - self.total_group[self.total_label[index]]
            img_path = os.path.join(self.data_root, self.ims[index])
            img = imageio.imread(img_path).astype(np.float32) / 255.
            msk, orig_msk = self.get_mask(index)

            H, W = img.shape[:2]
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)

            cam_ind = self.cam_inds[index]
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
                i = os.path.basename(img_path).split('.')[0]
                frame_index = int(i)
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

            wpts, ppts, A, big_A, pbw, Rh, Th = self.prepare_input(i)

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
            #转成原图坐标系
            ori_img = copy.deepcopy(img)
            ori_msk = copy.deepcopy(msk)

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

        elif self.human_part[self.total_label[index]] == 'face':
            # face part
            index = index - self.total_group[self.total_label[index]]
            img_path = self.face_ims[index]
            img = imageio.imread(img_path).astype(np.float32) / 255.
            msk_path = img_path.replace('images', 'masks')
            if not os.path.exists(msk_path):
                msk_path = img_path.replace('images', 'masks').replace('jpg', 'png')
            orig_msk = cv2.imread(msk_path)[..., 0].astype(np.float32)
            # orig_msk = (np.array(img).mean(axis=-1) != 1).astype(np.float32)
            msk = orig_msk.copy()

            H, W = img.shape[:2]
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            cam = self.face_cameras[index]
            # cam_ind = self.cam_inds[index]
            K = cam['K'][0][:3, :3]
            D = np.zeros([5, 1])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)

            orig_msk = cv2.undistort(orig_msk, K, D)

            R = cam['R'][0]
            T = cam['T'][0]

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

            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

            # read v_shaped
            # vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
            # tpose = np.load(vertices_path).astype(np.float32)
            # import ipdb; ipdb.set_trace(context=11)
            # from lib.utils.debugger import dbg
            # dbg.showL3D([tpose, self.face_tvertices])
            tbounds = if_nerf_dutils.get_face_bounds(self.face_tvertices)
            wbounds = if_nerf_dutils.get_face_bounds(self.face_tvertices)
            if False:
                # from IPython import embed;embed()
                vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
                tpose = np.load(vertices_path).astype(np.float32)
                wpts_c = (self.face_tvertices @ R.T + T.T) @ K.T
                wpts_c_b = (tpose @ R.T + T.T) @ K.T
                project_p_b = wpts_c_b[:, :2] / wpts_c_b[:, 2:]
                project_p = wpts_c[:, :2] / wpts_c[:, 2:]
                import ipdb;
                ipdb.set_trace(context=11)
                import matplotlib.pylab as plt;
                plt.figure();
                plt.imshow(img);
                plt.plot(project_p[:, 0], project_p[:, 1], 'r*');
                plt.show()
                import matplotlib.pylab as plt;
                plt.figure();
                plt.imshow(img);
                plt.plot(project_p_b[:, 0], project_p_b[:, 1], 'r*');
                plt.show()

            pose = np.concatenate([R, T], axis=1)
            bound_mask = if_nerf_dutils.get_bound_2d_mask(wbounds, K, pose, H, W)
            img, msk, K, crop_bbox = if_nerf_dutils.crop_image_msk(
                img, msk, K, bound_mask)
            H, W = img.shape[:2]
            pytorch3d_K = set_pytorch3d_intrinsic_matrix(K, H, W)
            ray_o, ray_d, near, far, mask_at_box = if_nerf_dutils.get_rays_within_bounds_test(
                H, W, K, R, T, wbounds)
            rgb = img[mask_at_box == 1]
            img = img.transpose(2, 0, 1)
            msk = (msk == 1).astype(np.uint8)
            smpl_R = np.eye(3).astype(np.float32)

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
                'cam_R': R,
                'cam_T': T,
                'pytorch3d_K': pytorch3d_K
            }

            # blend weight
            meta = {
                'face_dist': self.face_distance,
                'wbounds': wbounds,
                'tbounds': tbounds
            }
            ret.update(meta)

            # transformation
            meta = {'H': H, 'W': W}
            ret.update(meta)

            latent_index = ori_index
            # if cfg.test_novel_pose:
            #     latent_index = cfg.num_train_frame - 1
            meta = {
                'H': H,
                'W': W,
                'orig_H': orig_H,
                'orig_W': orig_W,
                'crop_bbox': crop_bbox,
                'latent_index': latent_index,
                'frame_index': frame_index,
                'part_type': 1

            }
            ret.update(meta)
        elif self.human_part[self.total_label[index]] == 'handl':
            # hand left
            index = index - self.total_group[self.total_label[index]]
            img_path = os.path.join(self.handl_data_root, self.handl_ims[index])
            img = imageio.imread(img_path).astype(np.float32) / 255.
            msk, orig_msk = self.get_mask(index, 'handl')

            H, W = img.shape[:2]
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)

            cam_ind = self.handl_cam_inds[index]
            K = np.array(self.handl_cams['K'][cam_ind])
            D = np.array(self.handl_cams['D'][cam_ind])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            orig_msk = cv2.undistort(orig_msk, K, D)

            R = np.array(self.handl_cams['R'][cam_ind])
            T = np.array(self.handl_cams['T'][cam_ind])

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

            i = os.path.basename(img_path).split('.')[0]
            frame_index = i

            # read v_shaped
            vertices_path = os.path.join(self.handl_lbs_root, 'bigpose_vertices.npy')
            tpose = np.load(vertices_path).astype(np.float32)
            # import ipdb; ipdb.set_trace(context=11)
            # from lib.utils.debugger import dbg
            # dbg.showL3D([tpose])
            tbounds = if_nerf_dutils.get_bounds(tpose, 'hand')
            tbw = np.load(os.path.join(self.handl_lbs_root, 'bigpose_bw.npy'))
            tbw = tbw.astype(np.float32)

            wpts, ppts, A, big_A, pbw, Rh, Th = self.prepare_input(i, 'handl')

            # debug the vertices and img
            if False:
                wpts_c = (wpts @ R.T + T.T) @ K.T
                project_p = wpts_c[:, :2] / wpts_c[:, 2:]
                import matplotlib.pylab as plt;
                plt.figure();
                plt.imshow(img);
                plt.plot(project_p[:, 0], project_p[:, 1], 'r*');
                plt.show()
                import ipdb;
                ipdb.set_trace(context=11)
                import matplotlib.pylab as plt;
                plt.figure();
                plt.imshow(img);
                plt.show()
            pbounds = if_nerf_dutils.get_bounds(ppts, 'hand')
            wbounds = if_nerf_dutils.get_bounds(wpts, 'hand')

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

            if cfg.train_with_normal:
                # normal is extracted from undistroted image
                normal = self.get_normal(index)
                normal = cv2.resize(normal, (W, H),
                                    interpolation=cv2.INTER_NEAREST)
                normal = normal[coord[:, 0], coord[:, 1]].astype(np.float32)
                ret.update({'normal': normal})

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

            # transformation
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
            ret.update(meta)

            # latent_index = index // self.num_cams + (cfg.begin_ith_frame//cfg.frame_interval)
            latent_index = index // self.handl_num_cams + cfg.num_train_frame + cfg.face_num_train_frame  # + (cfg.begin_ith_frame//cfg.frame_interval)
            if cfg.test_novel_pose:
                latent_index = cfg.num_train_frame - 1
            meta = {
                'H': H,
                'W': W,
                'orig_H': orig_H,
                'orig_W': orig_W,
                'crop_bbox': crop_bbox,
                'latent_index': latent_index,
                'frame_index': int(frame_index),
                'cam_ind': cam_ind,
                'part_type': 2
            }
            ret.update({'meta': {
                'frame_index': frame_index
            }})
            ret.update(meta)
        elif self.human_part[self.total_label[index]] == 'handr':
            # hand left
            index = index - self.total_group[self.total_label[index]]
            img_path = os.path.join(self.handr_data_root, self.handr_ims[index])
            img = imageio.imread(img_path).astype(np.float32) / 255.
            msk, orig_msk = self.get_mask(index, 'handr')

            H, W = img.shape[:2]
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)

            cam_ind = self.handr_cam_inds[index]
            K = np.array(self.handr_cams['K'][cam_ind])
            D = np.array(self.handr_cams['D'][cam_ind])
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            orig_msk = cv2.undistort(orig_msk, K, D)

            R = np.array(self.handr_cams['R'][cam_ind])
            T = np.array(self.handr_cams['T'][cam_ind])

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

            i = os.path.basename(img_path).split('.')[0]
            frame_index = i

            # read v_shaped
            vertices_path = os.path.join(self.handr_lbs_root, 'bigpose_vertices.npy')
            tpose = np.load(vertices_path).astype(np.float32)
            # import ipdb; ipdb.set_trace(context=11)
            # from lib.utils.debugger import dbg
            # dbg.showL3D([tpose])
            tbounds = if_nerf_dutils.get_bounds(tpose, 'hand')
            tbw = np.load(os.path.join(self.handr_lbs_root, 'bigpose_bw.npy'))
            tbw = tbw.astype(np.float32)

            wpts, ppts, A, big_A, pbw, Rh, Th = self.prepare_input(i, 'handr')

            # debug the vertices and img
            if False:
                wpts_c = (wpts @ R.T + T.T) @ K.T
                project_p = wpts_c[:, :2] / wpts_c[:, 2:]
                import matplotlib.pylab as plt;
                plt.figure();
                plt.imshow(img);
                plt.plot(project_p[:, 0], project_p[:, 1], 'r*');
                plt.show()
                import ipdb;
                ipdb.set_trace(context=11)
                import matplotlib.pylab as plt;
                plt.figure();
                plt.imshow(img);
                plt.show()
            pbounds = if_nerf_dutils.get_bounds(ppts, 'hand')
            wbounds = if_nerf_dutils.get_bounds(wpts, 'hand')

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

            if cfg.train_with_normal:
                # normal is extracted from undistroted image
                normal = self.get_normal(index)
                normal = cv2.resize(normal, (W, H),
                                    interpolation=cv2.INTER_NEAREST)
                normal = normal[coord[:, 0], coord[:, 1]].astype(np.float32)
                ret.update({'normal': normal})

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

            # transformation
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
            ret.update(meta)

            # latent_index = index // self.num_cams + (cfg.begin_ith_frame//cfg.frame_interval)
            latent_index = index // self.handl_num_cams + cfg.num_train_frame + cfg.face_num_train_frame  # + (cfg.begin_ith_frame//cfg.frame_interval)
            if cfg.test_novel_pose:
                latent_index = cfg.num_train_frame - 1
            meta = {
                'H': H,
                'W': W,
                'orig_H': orig_H,
                'orig_W': orig_W,
                'crop_bbox': crop_bbox,
                'latent_index': latent_index,
                'frame_index': int(frame_index),
                'cam_ind': cam_ind,
                'part_type': 3
            }
            ret.update({'meta': {
                'frame_index': frame_index
            }})
            ret.update(meta)

        vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        ret.update({'tpose': tpose})
        ret.update({'tbounds_body': self.tbounds_body,
                    'tbounds_face': self.tbounds_face,
                    'tbounds_handl': self.tbounds_handl,
                    'tbounds_handr': self.tbounds_handr,
                    'face_label': self.face_label,
                    'vertex_label': self.vertex_label})
        if cfg.train_with_overlap:
            # sample in the overlap region
            pts_bhl_overlap = self.get_sampling_points(self.tbounds_body_handl, N_samples=1024 * 4)
            # face
            pts_bf_overlap = self.get_sampling_points(self.tbounds_body_face_big, N_samples=1024 * 32)
            face_only_ind = self.get_inside(pts_bf_overlap, self.tbounds_body_face_small)
            pts_bf_overlap = pts_bf_overlap[~face_only_ind]
            # dbg.showL3D([tpose[::10], pts_bhl_overlap[::10], pts_bf_overlap[::10]])
            ret.update({'pts_bhl_overlap': pts_bhl_overlap,
                        'pts_bf_overlap': pts_bf_overlap})
        ret.update({'total_group_fix': np.array(self.total_group_fix)})
        if cfg.view_dirs_statics or cfg.face_vis_statics:
            if ori_index == len(self.total_ims) - 1:
                last_frame = 1
            else:
                last_frame = 0
            ret.update({'last_frame': last_frame})
        if cfg.debug.transform_face:
            ret.update({'T_canonical2face': - self.T_face2canonical@self.R_face2canonical,
                        'R_canonical2face': self.R_face2canonical.T,
                        'R_face2canonical': self.R_face2canonical,
                        'T_face2canonical': self.T_face2canonical})
        return ret

    def __len__(self):
        if cfg.vis_train_view:
            return cfg.vis_train_ni
        return len(self.total_ims)
