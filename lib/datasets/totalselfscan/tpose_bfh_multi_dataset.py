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

from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils.debugger import dbg


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
            #TODO:目前不支持novel pose
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame

        self.handl_ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.handl_cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
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

        self.handr_ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.handr_cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
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
        self.total_group_fix = [0,
                            cfg.fix_num_train_frame,
                            cfg.fix_num_train_frame+cfg.fix_face_num_train_frame,
                            cfg.fix_num_train_frame+cfg.fix_face_num_train_frame+cfg.fix_handl_num_train_frame]

        if False:
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
            tpose_body = np.load(vertices_path).astype(np.float32)
            vertices_path = os.path.join(self.handl_lbs_root, 'bigpose_vertices.npy')
            tpose_handl = np.load(vertices_path).astype(np.float32)
            vertices_path = os.path.join(self.handr_lbs_root, 'bigpose_vertices.npy')
            tpose_handr = np.load(vertices_path).astype(np.float32)
            dbg.showL3D([tpose_body, tpose_handl, tpose_handr, self.face_tvertices])
            dbg.showL3D([self.joints, self.handl_joints])

            import ipdb; ipdb.set_trace(context=11)
            dbg.showL3D([self.joints, self.handl_joints])

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
        tbounds_handl[0, 0] += cfg.data.hand_bound_reduce
        tbounds_handr[1, 0] -= cfg.data.hand_bound_reduce
        # deal with face bbox
        tbounds_face[0, 1] += cfg.data.face_bound_reduce
        # 减去left hand的space,交叉0.01m
        tbounds_body[1, 0] = tbounds_handl[0, 0] + cfg.data.body_hand_bound_overlap
        # 减去right hand的space,交叉0.01m
        tbounds_body[0, 0] = tbounds_handr[1, 0] - cfg.data.body_hand_bound_overlap
        # 减去face的space,交叉0.03m
        tbounds_body[1, 1] = tbounds_face[0, 1] + cfg.data.body_face_bound_overlap
        # dbg.showL3D([body_verts[::10], face_verts, handl_verts, tbounds_handl, tbounds_face, tbounds_body], lim=False)
        self.tbounds_body = tbounds_body
        self.tbounds_face = tbounds_face
        self.tbounds_handl = tbounds_handl
        self.tbounds_handr = tbounds_handr
        self.get_face_part_label()

    def get_face_part_label(self):
        # gt smplh mesh
        # tpose_dir = 'data/animation/{}/tpose_mesh.npy'.format(cfg.exp_name)
        # tpose_mesh = np.load(tpose_dir, allow_pickle=True).item()
        smplh_vertex_path = osp.join(self.data_root, 'lbs/bigpose_vertices.npy')
        smplh_face_path = osp.join(self.data_root, 'lbs/faces.npy')
        if not os.path.exists(smplh_vertex_path):
            smplh_vertex_path = osp.join(os.path.dirname(self.data_root), 'lbs/bigpose_vertices.npy')
            smplh_face_path = osp.join(os.path.dirname(self.data_root), 'lbs/faces.npy')
        smplh_vertex = np.load(smplh_vertex_path)
        smplh_face = np.load(smplh_face_path)
        self.tvertex = smplh_vertex
        self.tface = smplh_face.astype(np.int32)
        self.vertex_label = np.zeros(self.tvertex.shape[0])
        # part face
        pts_of_face = self.get_inside(self.tvertex, self.tbounds_face)
        self.vertex_label[pts_of_face] = 1
        face_of_face = pts_of_face[self.tface]
        face_of_face = face_of_face.sum(axis=1) == 3
        # part handl
        pts_of_handl = self.get_inside(self.tvertex, self.tbounds_handl)
        self.vertex_label[pts_of_handl] = 2
        face_of_handl = pts_of_handl[self.tface]
        face_of_handl = face_of_handl.sum(axis=1) == 3
        # part handr
        pts_of_handr = self.get_inside(self.tvertex, self.tbounds_handr)
        self.vertex_label[pts_of_handr] = 3
        face_of_handr = pts_of_handr[self.tface]
        face_of_handr = face_of_handr.sum(axis=1) == 3
        face_of_body = (~face_of_handl) * (~face_of_handr)
        face_label = np.zeros(self.tface.shape[0])
        face_label[face_of_face] = 1
        face_label[face_of_handl] = 2
        face_label[face_of_handr] = 3
        self.face_label = face_label

    def get_pre_face_part_label(self):
        tpose_dir = 'data/animation/{}/tpose_mesh.npy'.format(cfg.exp_name)
        tpose_mesh = np.load(tpose_dir, allow_pickle=True).item()
        self.pre_tvertex = tpose_mesh['vertex']
        self.pre_tface = tpose_mesh['triangle'].astype(np.int32)
        self.pre_vertex_label = np.zeros(self.pre_tvertex.shape[0])
        # part face
        pts_of_face = self.get_inside(self.pre_tvertex, self.tbounds_face)
        self.pre_vertex_label[pts_of_face] = 1
        face_of_face = pts_of_face[self.pre_tface]
        face_of_face = face_of_face.sum(axis=1) == 3
        # part handl
        pts_of_handl = self.get_inside(self.pre_tvertex, self.tbounds_handl)
        self.pre_vertex_label[pts_of_handl] = 2
        face_of_handl = pts_of_handl[self.pre_tface]
        face_of_handl = face_of_handl.sum(axis=1) == 3
        # part handr
        pts_of_handr = self.get_inside(self.pre_tvertex, self.tbounds_handr)
        self.pre_vertex_label[pts_of_handr] = 3
        face_of_handr = pts_of_handr[self.pre_tface]
        face_of_handr = face_of_handr.sum(axis=1) == 3
        face_of_body = (~face_of_handl) * (~face_of_handr)
        face_label = np.zeros(self.pre_tface.shape[0])
        face_label[face_of_face] = 1
        face_label[face_of_handl] = 2
        face_label[face_of_handr] = 3
        self.pre_face_label = face_label

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
            # Rh += np.random.rand(Rh.shape[0], Rh.shape[1])/2
            # prepare sp input of param pose
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            pxyz = np.dot(wxyz - Th, R).astype(np.float32)

            # calculate the skeleton transformation
            poses = params['poses'].reshape(-1, 3)
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

            if False:
                # from IPython import embed;embed()
                wpts_c = (wpts @ R.T + T.T) @ K.T
                project_p = wpts_c[:, :2] / wpts_c[:, 2:]
                import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(project_p[:, 0], project_p[:, 1], 'r*');plt.show()
                import ipdb; ipdb.set_trace(context=11)

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
                'part_type': 0

            }
            ret.update(meta)

        elif self.human_part[self.total_label[index]] == 'face':
            # face part
            index = index - self.total_group[self.total_label[index]]
            img_path = self.face_ims[index]
            img = imageio.imread(img_path).astype(np.float32) / 255.
            msk_path = img_path.replace('images', 'masks').replace('jpg', 'png')
            if not osp.exists(msk_path):
                msk_path = img_path.replace('images', 'masks')
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
            tbounds = if_nerf_dutils.get_face_bounds(self.face_tvertices)
            wbounds = if_nerf_dutils.get_face_bounds(self.face_tvertices)
            if False:
                # from IPython import embed;embed()
                # vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
                tpose = self.face_tvertices
                wpts_c = (self.face_tvertices @ R.T + T.T) @ K.T
                # wpts_c_b = (tpose @ R.T + T.T) @ K.T
                # project_p_b = wpts_c_b[:, :2] / wpts_c_b[:, 2:]
                project_p = wpts_c[:, :2] / wpts_c[:, 2:]
                import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(project_p[:, 0], project_p[:, 1], 'r*');plt.show()
                import ipdb; ipdb.set_trace(context=11)
                # import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(project_p_b[:, 0], project_p_b[:, 1], 'r*');plt.show()

            rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, wbounds, self.nrays, self.split)

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
                # import ipdb; ipdb.set_trace(context=11)
                # from lib.utils.debugger import dbg
                # dbg.showL3D([ppts, tpose])
                # import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.show()
            pbounds = if_nerf_dutils.get_bounds(ppts, 'hand')
            wbounds = if_nerf_dutils.get_bounds(wpts, 'hand')
            #TODO: hand mask sample有问题，只采一边的点
            rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, wbounds, self.nrays, self.split)

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
            if True:
                wpts_c = (wpts @ R.T + T.T) @ K.T
                project_p = wpts_c[:, :2] / wpts_c[:, 2:]
                import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.plot(project_p[:, 0], project_p[:, 1], 'r*');plt.show()
                # import ipdb; ipdb.set_trace(context=11)
                # from lib.utils.debugger import dbg
                # dbg.showL3D([ppts, tpose])
                # import matplotlib.pylab as plt;plt.figure();plt.imshow(img);plt.show()
            pbounds = if_nerf_dutils.get_bounds(ppts, 'hand')
            wbounds = if_nerf_dutils.get_bounds(wpts, 'hand')

            rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, wbounds, self.nrays, self.split)

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
            # latent_index = index // self.handr_num_cams + cfg.num_train_frame + cfg.face_num_train_frame + cfg.handl_num_train_frame  # + (cfg.begin_ith_frame//cfg.frame_interval)
            latent_index = index // self.handr_num_cams + cfg.fix_num_train_frame + cfg.fix_face_num_train_frame + cfg.fix_handl_num_train_frame  # + (cfg.begin_ith_frame//cfg.frame_interval)
            if cfg.test_novel_pose:
                #TODO: render的时候需要的latent index要想一下
                latent_index = cfg.num_train_frame - 1
            meta = {
                'latent_index': latent_index,
                # 'frame_index': frame_index,
                'cam_ind': cam_ind,
                'part_type': 3
            }
            ret.update({'meta': {
                'frame_index': frame_index
            }})
            ret.update(meta)
        ret.update({'tbounds_body': self.tbounds_body,
                    'tbounds_face': self.tbounds_face,
                    'tbounds_handl': self.tbounds_handl,
                    'tbounds_handr': self.tbounds_handr,
                    'face_label': self.face_label})
        if cfg.train_with_overlap:
            # sample in the overlap region
            pts_bhl_overlap = self.get_sampling_points(self.tbounds_body_handl, N_samples=1024*4)
            pts_bhr_overlap = self.get_sampling_points(self.tbounds_body_handr, N_samples=1024*4)
            # face
            pts_bf_overlap = self.get_sampling_points(self.tbounds_body_face_big, N_samples=1024*32)
            face_only_ind = self.get_inside(pts_bf_overlap, self.tbounds_body_face_small)
            pts_bf_overlap = pts_bf_overlap[~face_only_ind]
            # dbg.showL3D([tpose[::10], pts_bhl_overlap[::10], pts_bf_overlap[::10]])
            ret.update({'pts_bhl_overlap': pts_bhl_overlap,
                        'pts_bhr_overlap': pts_bhr_overlap,
                        'pts_bf_overlap': pts_bf_overlap})
        return ret

    def __len__(self):
        if cfg.vis_train_view:
            return cfg.vis_train_ni
        return len(self.total_ims)
