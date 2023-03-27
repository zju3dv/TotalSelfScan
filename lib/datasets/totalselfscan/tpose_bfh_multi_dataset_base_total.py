import torch.utils.data as data
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

from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils.total_utils import get_inside



class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split
        self.human_part = ['body', 'face', 'handl', 'handr']
        # body
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        if len(cfg.test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        view = cfg.training_view if split == 'train' else test_view

        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame

        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.num_body = len(self.ims)
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.num_cams = len(view)

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        self.nrays = cfg.N_rand

        # face
        i = cfg.face_begin_ith_frame
        i_intv = cfg.face_frame_interval
        ni = cfg.face_num_train_frame
        self.face_root = osp.join(self.data_root, 'face')
        self.face_ims = sorted(glob.glob(osp.join(self.face_root, 'images/*')))[i:i + ni * i_intv:i_intv]
        self.face_tvertices = np.load(osp.join(self.face_root, 'tvertices_face.npy'))
        self.face_cameras = np.load(osp.join(self.face_root, 'new_cameras.npy'), allow_pickle=True)[i:i + ni * i_intv:i_intv]
        assert len(self.face_ims) == len(self.face_cameras)
        self.num_face = len(self.face_ims)
        self.face_distance = np.load(osp.join(self.face_root, 'distance.npy'))

        # hand
        # handl
        split = 'train'
        if not cfg.new_hand:
            self.handl_data_root = osp.join(self.data_root, 'hand/handl')
        else:
            self.handl_data_root = osp.join(self.data_root, 'hand/handl_new')
        ann_file = osp.join(self.handl_data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.handl_cams = annots['cams']
        num_cams = len(self.handl_cams['K'])
        if len(cfg.handl_test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.handl_training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.handl_test_view
        view = cfg.handl_training_view if split == 'train' else test_view

        i = cfg.handl_begin_ith_frame
        i_intv = cfg.handl_frame_interval
        ni = cfg.handl_num_train_frame
        if cfg.test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame
        bad_frame = osp.join(self.handl_data_root, 'frame_remove.json')
        select_frames = list(range(i, i + ni * i_intv, i_intv))
        if osp.exists(bad_frame):
            with open(bad_frame, 'r') as f:
                bad_frame = json.load(f)['bad_frame']
            bad_frame_list = []
            for frame_i in bad_frame:
                bad_frame_list += list(range(frame_i[0], frame_i[1]))
            bad_frame = bad_frame_list
            select_frames = [frame for frame in select_frames if frame not in bad_frame]

        self.handl_ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in np.array(annots['ims'])[select_frames].tolist()
        ]).ravel()
        self.handl_cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in np.array(annots['ims'])[select_frames].tolist()
        ]).ravel()
        # self.get_img(annots, view, 'handl')

        hand_repeat = cfg.hand_repeat
        self.handl_ims = np.repeat(self.handl_ims[None], hand_repeat, axis=0).reshape(-1)
        self.num_handl = len(self.handl_ims)
        self.handl_cam_inds = np.repeat(self.handl_cam_inds[None], hand_repeat, axis=0).reshape(-1)
        self.handl_num_cams = len(view)
        self.handl_lbs_root = os.path.join(self.handl_data_root, 'lbs')
        joints = np.load(os.path.join(self.handl_lbs_root, 'joints.npy'))
        self.handl_joints = joints.astype(np.float32)
        self.handl_parents = np.load(os.path.join(self.handl_lbs_root, 'parents.npy'))
        # handr
        split = 'train'
        if not cfg.new_hand:
            self.handr_data_root = osp.join(self.data_root, 'hand/handr')
        else:
            self.handr_data_root = osp.join(self.data_root, 'hand/handr_new')
        ann_file = osp.join(self.handr_data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.handr_cams = annots['cams']
        num_cams = len(self.handr_cams['K'])
        if len(cfg.handr_test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.handr_training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.handr_test_view
        view = cfg.handr_training_view if split == 'train' else test_view

        i = cfg.handr_begin_ith_frame
        i_intv = cfg.handr_frame_interval
        ni = cfg.handr_num_train_frame
        if cfg.test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame
        bad_frame = osp.join(self.handr_data_root, 'frame_remove.json')
        select_frames = list(range(i, i + ni * i_intv, i_intv))
        if osp.exists(bad_frame):
            with open(bad_frame, 'r') as f:
                bad_frame = json.load(f)['bad_frame']
            bad_frame_list = []
            for frame_i in bad_frame:
                bad_frame_list += list(range(frame_i[0], frame_i[1]))
            bad_frame = bad_frame_list
            self.bad_frame = bad_frame_list
            select_frames = [frame for frame in select_frames if frame not in bad_frame]

        self.handr_ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in np.array(annots['ims'])[select_frames].tolist()
        ]).ravel()
        self.handr_cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in np.array(annots['ims'])[select_frames].tolist()
        ]).ravel()
        # self.get_img(annots, view, 'handr')

        hand_repeat = cfg.hand_repeat
        self.handr_ims = np.repeat(self.handr_ims[None], hand_repeat, axis=0).reshape(-1)
        self.handr_cam_inds = np.repeat(self.handr_cam_inds[None], hand_repeat, axis=0).reshape(-1)
        self.handr_num_cams = len(view)
        self.handr_lbs_root = os.path.join(self.handr_data_root, 'lbs')
        joints = np.load(os.path.join(self.handr_lbs_root, 'joints.npy'))
        self.handr_joints = joints.astype(np.float32)
        self.handr_parents = np.load(os.path.join(self.handr_lbs_root, 'parents.npy'))
        # total human
        self.total_ims = np.concatenate([self.ims, self.face_ims, self.handl_ims, self.handr_ims])
        self.total_label = np.concatenate([
            np.zeros(len(self.ims)).astype(np.int32),
            np.ones(len(self.face_ims)).astype(np.int32),
            (np.ones(len(self.handl_ims)) * 2).astype(np.int32),
            (np.ones(len(self.handr_ims)) * 3).astype(np.int32)
        ])
        self.total_group = [0,
                            self.num_body,
                            self.num_body+self.num_face,
                            self.num_body+self.num_face+self.num_handl]

        # obtain the bbox for each part in canonical space, i.e., body, face, hand, bodyface, bodyhand
        body_vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        body_verts = np.load(body_vertices_path).astype(np.float32)
        face_verts = self.face_tvertices
        handl_vertices_path = os.path.join(self.handl_lbs_root, 'bigpose_vertices.npy')
        handl_verts = np.load(handl_vertices_path).astype(np.float32)
        handr_vertices_path = os.path.join(self.handr_lbs_root, 'bigpose_vertices.npy')
        handr_verts = np.load(handr_vertices_path).astype(np.float32)
        tbounds_handl = if_nerf_dutils.get_bounds(handl_verts, 'hand')
        tbounds_handr = if_nerf_dutils.get_bounds(handr_verts, 'hand')
        tbounds_face = if_nerf_dutils.get_bounds(face_verts, 'face')
        tbounds_body = if_nerf_dutils.get_bounds(body_verts, 'body')
        # deal with hand bbox
        tbounds_handl[0, 0] += cfg.data.hand_bound_reduce #0.03
        tbounds_handr[1, 0] -= cfg.data.hand_bound_reduce #0.03
        # deal with face bbox
        tbounds_face[0, 1] += cfg.data.face_bound_reduce
        # overlap 0.01m
        tbounds_body[1, 0] = tbounds_handl[0, 0] + cfg.data.body_hand_bound_overlap #0.02
        # overlap 0.01m
        tbounds_body[0, 0] = tbounds_handr[1, 0] - cfg.data.body_hand_bound_overlap #0.02
        # overlap 0.03m
        tbounds_body[1, 1] = tbounds_face[0, 1] + cfg.data.body_face_bound_overlap
        self.tbounds_body = tbounds_body
        self.tbounds_face = tbounds_face
        self.tbounds_handl = tbounds_handl
        self.tbounds_handr = tbounds_handr

        # load handl mesh
        if cfg.init_handl != 'no':
            handl_mesh = np.load(osp.join('data/animation', cfg.init_handl, 'tpose_mesh.npy'), allow_pickle=True).item()
            handl_vertex = handl_mesh['vertex']
            idx_in_body = get_inside(handl_vertex, self.tbounds_body)
            idx_in_handl = get_inside(handl_vertex, self.tbounds_handl)
            idx_body_handl_both = idx_in_body * idx_in_handl
            self.body_handl_pts = handl_vertex[idx_body_handl_both].astype(np.float32)
            samples_idx = np.random.randint(0, self.body_handl_pts.shape[0], 3000)
            self.body_handl_pts = self.body_handl_pts[samples_idx]
        # load handr mesh
        if cfg.init_handr != 'no':
            handr_mesh = np.load(osp.join('data/animation', cfg.init_handr, 'tpose_mesh.npy'), allow_pickle=True).item()
            handr_vertex = handr_mesh['vertex']
            idx_in_body = get_inside(handr_vertex, self.tbounds_body)
            idx_in_handr = get_inside(handr_vertex, self.tbounds_handr)
            idx_body_handr_both = idx_in_body * idx_in_handr
            self.body_handr_pts = handr_vertex[idx_body_handr_both].astype(np.float32)
            samples_idx = np.random.randint(0, self.body_handr_pts.shape[0], 3000)
            self.body_handr_pts = self.body_handr_pts[samples_idx]
        # load face mesh
        self.transform_face_coord = False
        if cfg.init_face != 'no':
            face_mesh = np.load(osp.join('data/animation', cfg.init_face, 'tpose_mesh.npy'), allow_pickle=True).item()
            face_vertex = face_mesh['vertex']
            face_coord_transform_file = osp.join('data/animation', cfg.init_face, 'transform.npy')
            if osp.exists(face_coord_transform_file):
                face_coord_transform = np.load(face_coord_transform_file, allow_pickle=True).item()
                R_face2canonical = face_coord_transform['R_f'][0]
                T_face2canonical = face_coord_transform['T_f']
                face_verext_transformed = face_vertex @ R_face2canonical.T + T_face2canonical
                face_vertex = face_verext_transformed
                self.transform_face_coord = True
                self.R_face2canonical = R_face2canonical
                self.T_face2canonical = T_face2canonical

            idx_in_body = get_inside(face_vertex, self.tbounds_body)
            idx_in_face = get_inside(face_vertex, self.tbounds_face)
            idx_body_face_both = idx_in_body * idx_in_face
            self.body_face_pts = face_vertex[idx_body_face_both].astype(np.float32)
            samples_idx = np.random.randint(0, self.body_face_pts.shape[0], 3000)
            self.body_face_pts = self.body_face_pts[samples_idx]
            # from lib.utils.debugger import dbg
            # import ipdb; ipdb.set_trace(context=11)
            # dbg.showL3D([self.body_face_pts,face_vertex[::40]])

        # sample less view
        if cfg.debug.sample_less_view:
            self.face_vis = np.load(osp.join('data/face_vis', cfg.exp_name, 'face_vis.npy'), allow_pickle=True).item()
            self.face_need_sample = [key for key,val in self.face_vis[0].items() if val['num'] <= 3]
            self.face_vis_need_sample = {key: val for key,val in self.face_vis[0].items() if val['num'] <= 3}
    def get_img(self, annots, view, part):
        if part == 'handl':
            vertice_path = osp.join(self.handl_data_root, 'vertices')
            files = sorted(os.listdir(vertice_path))
            vert_idx = np.array([int(file.split('.')[0]) for file in files])
            self.fix_handl_num_train_frame = len(files)
            self.handl_ims = np.array([
                np.array(ims_data['ims'])[view]
                for ims_data in np.array(annots['ims'])[vert_idx]
            ]).ravel()

            self.handl_cam_inds = np.array([
                np.arange(len(ims_data['ims']))[view]
                for ims_data in np.array(annots['ims'])[vert_idx]
            ]).ravel()
        elif part == 'handr':
            vertice_path = osp.join(self.handr_data_root, 'vertices')
            files = sorted(os.listdir(vertice_path))
            vert_idx = np.array([int(file.split('.')[0]) for file in files])
            self.fix_handr_num_train_frame = len(files)
            self.handr_ims = np.array([
                np.array(ims_data['ims'])[view]
                for ims_data in np.array(annots['ims'])[vert_idx]
            ]).ravel()

            self.handr_cam_inds = np.array([
                np.arange(len(ims_data['ims']))[view]
                for ims_data in np.array(annots['ims'])[vert_idx]
            ]).ravel()

    def get_mask(self, index, part='body'):
        if part == 'body':
            msk_path = os.path.join(self.data_root, 'mask_cihp',
                                    self.ims[index])[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.data_root, self.ims[index].replace(
                    'images', 'mask'))[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.data_root, self.ims[index].replace(
                    'images', 'mask'))[:-4] + '.jpg'
            msk_cihp = imageio.imread(msk_path)
            if len(msk_cihp.shape) == 3:
                msk_cihp = msk_cihp[..., 0]
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
            msk = msk_cihp
            orig_msk = msk.copy()


            if (not cfg.eval and cfg.erode_edge):
                border = 5
                kernel = np.ones((border, border), np.uint8)
                msk_erode = cv2.erode(msk.copy(), kernel)
                msk_dilate = cv2.dilate(msk.copy(), kernel)
                msk[(msk_dilate - msk_erode) == 1] = 100
            if cfg.vis_train_view:
                border = 120
                kernel = np.ones((border, border), np.uint8)
                msk_erode = cv2.erode(msk.copy(), kernel)
                msk_dilate = cv2.dilate(msk.copy(), kernel)
                msk = msk_dilate
        elif part == 'handl':
            msk_path = os.path.join(self.handl_data_root, 'mask_cihp',
                                    self.handl_ims[index])[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.handl_data_root, self.handl_ims[index].replace(
                    'images', 'mask'))[:-4] + '.png'
            if not osp.exists(msk_path):
                msk_path = msk_path.replace('png', 'jpg')
            msk_cihp = imageio.imread(msk_path)
            if len(msk_cihp.shape) == 3:
                msk_cihp = msk_cihp[..., 0]
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
            msk = msk_cihp
            semantic_path = msk_path.replace('mask', 'semantic').replace('jpg', 'npy')
            if osp.exists(semantic_path):
                sematic_msk = (np.load(semantic_path)*255).astype(np.uint8)
                border_erode = 10
                border_dilate = 40
                kernel_erode = np.ones((border_erode, border_erode), np.uint8)
                kernel_dilate = np.ones((border_dilate, border_dilate), np.uint8)
                msk_erode = cv2.erode(sematic_msk.copy(), kernel_erode)
                msk_dilate = cv2.dilate(msk_erode, kernel_dilate)
                msk_dilate[msk_dilate==255] = 1
                msk = msk * msk_dilate
            orig_msk = msk.copy()
        elif part == 'handr':
            msk_path = os.path.join(self.handr_data_root, 'mask_cihp',
                                    self.handr_ims[index])[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.handr_data_root, self.handr_ims[index].replace(
                    'images', 'mask'))[:-4] + '.png'
            if not osp.exists(msk_path):
                msk_path = msk_path.replace('png', 'jpg')
            msk_cihp = imageio.imread(msk_path)
            if len(msk_cihp.shape) == 3:
                msk_cihp = msk_cihp[..., 0]
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
            msk = msk_cihp
            semantic_path = msk_path.replace('mask', 'semantic').replace('jpg', 'npy')
            if osp.exists(semantic_path):
                sematic_msk = (np.load(semantic_path)*255).astype(np.uint8)
                border_erode = 10
                border_dilate = 40
                kernel_erode = np.ones((border_erode, border_erode), np.uint8)
                kernel_dilate = np.ones((border_dilate, border_dilate), np.uint8)
                msk_erode = cv2.erode(sematic_msk.copy(), kernel_erode)
                msk_dilate = cv2.dilate(msk_erode, kernel_dilate)
                msk_dilate[msk_dilate==255] = 1
                msk = msk * msk_dilate
            orig_msk = msk.copy()

        return msk, orig_msk

    def get_normal(self, index):
        normal_path = os.path.join(self.data_root, 'normal',
                                   self.ims[index])[:-4] + '.png'
        normal = imageio.imread(normal_path) / 255.
        normal = normal * 2 - 1
        return normal

    def prepare_input(self, i, part='body'):
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

            # prepare sp input of param pose
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            pxyz = np.dot(wxyz - Th, R).astype(np.float32)

            # calculate the skeleton transformation
            poses = params['poses'].reshape(-1, 3)
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

    def get_sampling_points(self, bounds, N_samples):
        min_xyz = bounds[0]
        max_xyz = bounds[1]
        x_vals = np.random.rand(1, N_samples)
        y_vals = np.random.rand(1, N_samples)
        z_vals = np.random.rand(1, N_samples)
        vals = np.concatenate([x_vals, y_vals, z_vals], axis=0)
        pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]

        return pts.T
    def get_inside(self, pts, bound):
        inside = pts > bound[:1]
        inside = inside * (pts < bound[1:])
        inside = np.sum(inside, axis=1) == 3

        return inside

    def __getitem__(self, index):
        if cfg.final_hand:
            # increase the portion of hand image
            if self.total_label[index] <= 1:
                # if np.random.rand() <= 0.5:
                index = self.total_group[2] + np.random.randint(len(self.handl_ims) + len(self.handr_ims))
        if cfg.vis_train_view:
            index = cfg.vis_train_begin_i + index * cfg.vis_train_interval

        ori_index = index
        if self.human_part[self.total_label[index]] == 'body':
            # body part
            self.part = 'body'
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
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            if cfg.mask_bkgd:
                img[msk == 0] = 0
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


            if cfg.train_with_coord and self.split == 'train':
                coord_path = os.path.join(
                    self.data_root,
                    'train_coord/frame_{:04d}_view_{:04d}.npy'.format(
                        frame_index, cam_ind))
                train_coord = np.load(coord_path, allow_pickle=True).item()
                rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_coord(
                    img, msk, train_coord, K, R, T, wbounds, self.nrays)
            else:
                rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                    img, msk, K, R, T, wbounds, self.nrays, self.split)
                if cfg.debug.sample_less_view:
                    ray_o_, ray_d_, near_, far_, mask_at_box_ = if_nerf_dutils.get_rays_within_bounds_test(
                        H, W, K, R, T, wbounds)
                    new_near_ = np.zeros([H, W]).astype(np.float32)
                    new_near_[mask_at_box_] = near_
                    new_far_ = np.zeros([H, W]).astype(np.float32)
                    new_far_[mask_at_box_] = far_

                    coord_need_sample = []
                    for face_i in self.face_vis_need_sample.keys():
                        if index in self.face_vis_need_sample[face_i]['frame_coord'].keys():
                            coord_need_sample += self.face_vis_need_sample[face_i]['frame_coord'][index]
                    coord_need_sample = np.array(coord_need_sample) * cfg.ratio
                    coord_need_sample = coord_need_sample.astype(np.int)
                    rgb = np.concatenate([rgb, img[coord_need_sample[:, 0], coord_need_sample[:, 1]]])
                    ray_o = np.concatenate([ray_o, ray_o_[coord_need_sample[:, 0], coord_need_sample[:, 1]]])
                    ray_d = np.concatenate([ray_d, ray_d_[coord_need_sample[:, 0], coord_need_sample[:, 1]]])
                    mask_at_box = np.concatenate([mask_at_box, mask_at_box_[coord_need_sample[:, 0], coord_need_sample[:, 1]]])
                    near = np.concatenate([near, new_near_.reshape(H, W)[coord_need_sample[:, 0], coord_need_sample[:, 1]]])
                    far = np.concatenate([far, new_far_.reshape(H, W)[coord_need_sample[:, 0], coord_need_sample[:, 1]]])
                    coord = np.concatenate([coord, coord_need_sample])
                    # import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(coord_need_sample[:,1],coord_need_sample[:,0],'r+');plt.show()

            if cfg.erode_edge:
                orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
            occupancy = orig_msk[coord[:, 0], coord[:, 1]]

            # nerf
            ret = {
                'rgb': rgb,
                'occupancy': occupancy,
                'ray_o': ray_o,
                'ray_d': ray_d,
                'near': near,
                'far': far,
                'mask_at_box': mask_at_box
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
            latent_index = index // self.num_cams #+ (cfg.begin_ith_frame//cfg.frame_interval)
            if cfg.test_novel_pose:
                latent_index = cfg.num_train_frame - 1
            meta = {
                'latent_index': latent_index,
                'frame_index': frame_index,
                'cam_ind': cam_ind,
                'img_type': 0

            }
            ret.update(meta)

        elif self.human_part[self.total_label[index]] == 'face':
            # face part
            self.part = 'face'
            index = index - self.total_group[self.total_label[index]]
            img_path = self.face_ims[index]
            img = imageio.imread(img_path).astype(np.float32) / 255.
            msk_path = img_path.replace('images', 'masks').replace('jpg', 'png')
            if not osp.exists(msk_path):
                msk_path = img_path.replace('images', 'masks')
            orig_msk = cv2.imread(msk_path)[..., 0].astype(np.float32)
            render_msk_path = msk_path.replace('masks', 'render_masks')
            if osp.exists(render_msk_path):
                render_msk = cv2.imread(render_msk_path)[..., :1]
                position = np.argwhere(render_msk==255).max(axis=0)
                render_msk_seg = np.zeros_like(orig_msk)
                render_msk_seg[:position[0]] = 1
                orig_msk = orig_msk * render_msk_seg
            # orig_msk = (np.array(img).mean(axis=-1) != 1).astype(np.float32)
            msk = orig_msk.copy()

            H, W = img.shape[:2]
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            cam = copy.deepcopy(self.face_cameras[index])
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
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            if cfg.mask_bkgd:
                img[msk == 0] = 0
            K[:2] = K[:2] * cfg.ratio

            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

            # read v_shaped
            # vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
            # tpose = np.load(vertices_path).astype(np.float32)
            # import ipdb; ipdb.set_trace(context=11)
            # from lib.utils.debugger import dbg
            # dbg.showL3D([tpose, self.face_tvertices])
            tbounds = if_nerf_dutils.get_face_bounds(self.face_tvertices, delta=0.1)
            wbounds = if_nerf_dutils.get_face_bounds(self.face_tvertices, delta=0.1)
            if False:
                # from IPython import embed;embed()
                # vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
                tpose = self.face_tvertices
                wpts_c = (self.face_tvertices @ R.T + T.T) @ K.T
                # wpts_c_b = (tpose @ R.T + T.T) @ K.T
                # project_p_b = wpts_c_b[:, :2] / wpts_c_b[:, 2:]
                project_p = wpts_c[:, :2] / wpts_c[:, 2:]
                # import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.show()
                import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(project_p[:, 0], project_p[:, 1], 'r*');plt.show()
                # import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(project_p_b[:, 0], project_p_b[:, 1], 'r*');plt.show()

            rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, wbounds, self.nrays, self.split, cfg.debug.all_video_train)

            occupancy = orig_msk[coord[:, 0], coord[:, 1]]

            # nerf
            ret = {
                'rgb': rgb,
                'occupancy': occupancy,
                'ray_o': ray_o,
                'ray_d': ray_d,
                'near': near,
                'far': far,
                'mask_at_box': mask_at_box
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

            # latent_index = ori_index
            latent_index = index + cfg.fix_num_train_frame
            # if cfg.test_novel_pose:
            #     latent_index = cfg.num_train_frame - 1
            meta = {
                'latent_index': latent_index,
                'frame_index': frame_index,
                'img_type': 1
            }
            ret.update(meta)
        elif self.human_part[self.total_label[index]] == 'handl':
            # hand left
            self.part = 'hand'
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
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            if cfg.mask_bkgd:
                img[msk == 0] = 0
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
                import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(project_p[:, 0], project_p[:, 1], 'r*');plt.show()
                import matplotlib.pylab as plt;plt.figure();plt.imshow(msk);plt.show()
                # from lib.utils.debugger import dbg
                # dbg.showL3D([ppts, tpose])
                # import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.show()
            pbounds = if_nerf_dutils.get_bounds(ppts, 'hand')
            wbounds = if_nerf_dutils.get_bounds(wpts, 'hand')

            rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, wbounds, self.nrays, self.split, cfg.debug.all_video_train)

            if cfg.erode_edge:
                orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
            occupancy = orig_msk[coord[:, 0], coord[:, 1]]

            # nerf
            ret = {
                'rgb': rgb,
                'occupancy': occupancy,
                'ray_o': ray_o,
                'ray_d': ray_d,
                'near': near,
                'far': far,
                'mask_at_box': mask_at_box
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
            # latent_index = index // self.handl_num_cams + cfg.num_train_frame + cfg.face_num_train_frame  # + (cfg.begin_ith_frame//cfg.frame_interval)
            latent_index = index // self.handl_num_cams + cfg.fix_num_train_frame + cfg.fix_face_num_train_frame  # + (cfg.begin_ith_frame//cfg.frame_interval)
            if cfg.test_novel_pose:
                latent_index = cfg.num_train_frame - 1
            meta = {
                'latent_index': latent_index,
                # 'frame_index': frame_index,
                'cam_ind': cam_ind,
                'img_type': 2
            }
            ret.update({'meta': {
                'frame_index': frame_index
            }})
            ret.update(meta)
        elif self.human_part[self.total_label[index]] == 'handr':
            # hand left
            self.part = 'hand'
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
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            orig_msk = cv2.resize(orig_msk, (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            if cfg.mask_bkgd:
                img[msk == 0] = 0
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
            # if False and int(i) in self.bad_frame:
                wpts_c = (wpts @ R.T + T.T) @ K.T
                project_p = wpts_c[:, :2] / wpts_c[:, 2:]
                import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(project_p[:, 0], project_p[:, 1], 'r*');plt.show()
                # from lib.utils.debugger import dbg
                # dbg.showL3D([ppts, tpose])
                # import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.show()
            pbounds = if_nerf_dutils.get_bounds(ppts, 'hand')
            wbounds = if_nerf_dutils.get_bounds(wpts, 'hand')

            rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, wbounds, self.nrays, self.split, cfg.debug.all_video_train)

            if cfg.erode_edge:
                orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
            occupancy = orig_msk[coord[:, 0], coord[:, 1]]

            # nerf
            ret = {
                'rgb': rgb,
                'occupancy': occupancy,
                'ray_o': ray_o,
                'ray_d': ray_d,
                'near': near,
                'far': far,
                'mask_at_box': mask_at_box
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

            latent_index = index // self.handr_num_cams + cfg.fix_num_train_frame + cfg.fix_face_num_train_frame + cfg.fix_handl_num_train_frame  # + (cfg.begin_ith_frame//cfg.frame_interval)
            if cfg.test_novel_pose:
                latent_index = cfg.num_train_frame - 1
            meta = {
                'latent_index': latent_index,
                # 'frame_index': frame_index,
                'cam_ind': cam_ind,
                'img_type': 3
            }
            ret.update({'meta': {
                'frame_index': frame_index
            }})
            ret.update(meta)
        ret.update({'tbounds_body': self.tbounds_body,
                    'tbounds_face': self.tbounds_face,
                    'tbounds_handl': self.tbounds_handl,
                    'tbounds_handr': self.tbounds_handr})
        if 'meta' in ret.keys():
            ret['meta'].update({'transform_face_coord': self.transform_face_coord})
        else:
            ret['meta'] = {}
            ret['meta'].update({'transform_face_coord': self.transform_face_coord})
        if self.transform_face_coord:
            ret.update({'T_canonical2face': - self.T_face2canonical@self.R_face2canonical,
                        'R_canonical2face': self.R_face2canonical.T})

        if hasattr(self, 'body_handl_pts'):
            ret.update({'body_handl_pts': self.body_handl_pts})
        if hasattr(self, 'body_handr_pts'):
            ret.update({'body_handr_pts': self.body_handr_pts})
        if hasattr(self, 'body_face_pts'):
            ret.update({'body_face_pts': self.body_face_pts})
        if not cfg.fix_body:
            ret.update({'part_type': 0})
        if not cfg.fix_face:
            ret.update({'part_type': 1})
        if not cfg.fix_handl:
            ret.update({'part_type': 2})
        if not cfg.fix_handr:
            ret.update({'part_type': 3})

        return ret

    def __len__(self):
        if cfg.vis_train_view:
            return cfg.vis_train_ni
        return len(self.total_ims)
