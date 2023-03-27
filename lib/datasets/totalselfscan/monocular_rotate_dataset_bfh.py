import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import copy
import json
import os
import os.path as osp
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils import render_utils


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        test_view = [0]
        view = cfg.training_view if split == 'train' else test_view
        self.num_cams = len(view)
        K, RT = render_utils.load_cam(ann_file)

        i = 0
        # i = cfg.begin_ith_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][i:i + cfg.num_train_frame *
                                          cfg.frame_interval]
        ])

        self.K = K[0]

        self.Ks = np.array(K)[cfg.training_view].astype(np.float32)
        self.RT = np.array(RT)[cfg.training_view].astype(np.float32)
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(
            np.float32)

        # self.ts = [0]
        self.ts = np.arange(0, np.pi * 2, np.pi / 18)
        self.nt = len(self.ts)

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))

        # obtain the bbox for each part in canonical space, i.e., body, face, hand, bodyface, bodyhand
        self.face_root = osp.join(self.data_root, 'face')
        self.face_tvertices = np.load(osp.join(self.face_root, 'tvertices_face.npy'))
        body_vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        body_verts = np.load(body_vertices_path).astype(np.float32)
        face_verts = self.face_tvertices
        self.handl_data_root = osp.join(self.data_root, 'hand/handl_new')
        self.handl_lbs_root = os.path.join(self.handl_data_root, 'lbs')
        handl_vertices_path = os.path.join(self.handl_lbs_root, 'bigpose_vertices.npy')
        handl_verts = np.load(handl_vertices_path).astype(np.float32)
        tbounds_handl = if_nerf_dutils.get_bounds(handl_verts, 'hand')
        tbounds_face = if_nerf_dutils.get_bounds(face_verts, 'face')
        tbounds_body = if_nerf_dutils.get_bounds(body_verts, 'body')
        # 减去left hand的space,交叉0.01m
        tbounds_body[1, 0] = tbounds_handl[0, 0] + 0.02
        # 减去face的space,交叉0.03m
        tbounds_body[1, 1] = tbounds_face[0, 1] + 0.05
        # dbg.showL3D([body_verts[::10], face_verts, handl_verts, tbounds_handl, tbounds_face, tbounds_body], lim=False)
        self.tbounds_body = tbounds_body
        self.tbounds_face = tbounds_face
        self.tbounds_handl = tbounds_handl
        # 计算handl的交叉区域
        tbounds_body_handl = copy.deepcopy(tbounds_handl)
        tbounds_body_handl[0, 0] = tbounds_handl[0, 0] - 0.01
        tbounds_body_handl[1, 0] = tbounds_handl[0, 0] + 0.01
        self.tbounds_body_handl = tbounds_body_handl
        # 计算face的交叉区域
        tbounds_body_face = copy.deepcopy(tbounds_face)
        tbounds_body_face[0] -= 0.01
        tbounds_body_face[1] += 0.01
        tbounds_body_face[1, 1] = tbounds_body_face[0, 1] + 0.15
        self.tbounds_body_face_big = tbounds_body_face
        self.tbounds_body_face_small = tbounds_face

    def prepare_input(self, i, index):
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = i + 1

        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        t = self.ts[index]
        rot_ = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        rot = np.eye(3)
        rot[[0, 0, 2, 2], [0, 2, 0, 2]] = rot_.ravel()
        center = np.mean(wxyz, axis=0)
        wxyz = wxyz - center
        wxyz = np.dot(wxyz, rot.T)
        wxyz = wxyz + center
        wxyz = wxyz.astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        R = np.dot(rot, R)
        Rh = cv2.Rodrigues(R)[0]
        Th = np.sum(rot * (Th - center), axis=1) + center
        Th = Th.astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)
        big_poses = np.zeros_like(poses).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        # big_poses[23] = np.deg2rad(-angle)
        # big_poses[26] = np.deg2rad(angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, joints, parents)
        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, pbw, Rh, Th, big_A

    def get_mask(self, i):
        ims = self.ims[i]
        msks = []

        for nv in range(len(ims)):
            im = ims[nv]

            msk_path = os.path.join(self.data_root, 'mask_cihp',
                                    im)[:-4] + '.png'
            if not os.path.exists(msk_path):
                path = im.split('/')[1:]
                msk_path = os.path.join(self.data_root, 'mask',
                                        path[0], path[1])[:-4] + '.png'
            msk_cihp = imageio.imread(msk_path)
            msk_cihp = (msk_cihp != 0).astype(np.uint8)

            msk = msk_cihp.astype(np.uint8)

            K = self.Ks[nv].copy()
            K[:2] = K[:2] / cfg.ratio
            msk = cv2.undistort(msk, K, self.Ds[nv])

            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk = cv2.dilate(msk.copy(), kernel)

            msks.append(msk)

        return msks

    def __getitem__(self, index):
        latent_index = cfg.begin_ith_frame * cfg.frame_interval
        frame_index = cfg.begin_ith_frame * cfg.frame_interval

        msks = self.get_mask(frame_index)

        img_name = os.path.basename(self.ims[frame_index][0])[:-4]
        frame_index = img_name
        # read v_shaped
        vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        tbounds = if_nerf_dutils.get_bounds(tpose)
        tbw = np.load(os.path.join(self.lbs_root, 'bigpose_bw.npy'))
        tbw = tbw.astype(np.float32)

        wpts, ppts, A, pbw, Rh, Th, big_A = self.prepare_input(
            frame_index, index)

        pbounds = if_nerf_dutils.get_bounds(ppts)
        wbounds = if_nerf_dutils.get_bounds(wpts)


        # reduce the image resolution by ratio
        img_path = os.path.join(self.data_root, self.ims.ravel()[0])
        H, W = imageio.imread(img_path).shape[:2]
        H, W = int(H * cfg.ratio), int(W * cfg.ratio)
        msks = [
            cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            for msk in msks
        ]
        msks = np.array(msks)
        K = self.K

        view_index = 0
        K = self.Ks[view_index]
        RT = self.RT[view_index]
        R, T = RT[:3, :3], RT[:3, 3:]
        ray_o, ray_d, near, far, mask_at_box = if_nerf_dutils.get_rays_within_bounds(
            H, W, K, R, T, wbounds)
        # ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(
        #         RT, K, wbounds)

        ret = {
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
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

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': int(frame_index),
            'view_index': index
        }
        ret.update(meta)

        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(frame_index))
        params = np.load(params_path, allow_pickle=True).item()
        Rh0 = params['Rh'][:3]
        R0 = cv2.Rodrigues(Rh0)[0].astype(np.float32)
        Th0 = params['Th'][0].astype(np.float32)
        meta = {
            'msk': msks[0],
            'R0_snap': R0,
            'Th0_snap': Th0,
            'K': self.Ks[0],
            'RT': self.RT[0, :3],
            'H': H,
            'W': W,
            'part_type': 0
        }
        ret.update(meta)
        ret.update({'tbounds_body': self.tbounds_body,
                    'tbounds_face': self.tbounds_face,
                    'tbounds_handl': self.tbounds_handl})

        return ret

    def __len__(self):
        return self.nt
