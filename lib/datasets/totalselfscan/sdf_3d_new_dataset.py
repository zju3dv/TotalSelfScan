import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import os.path as osp
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
import trimesh
from . import tpose_dataset


class Dataset(tpose_dataset.Dataset):
    def __init__(self, data_root, human, ann_file, split, **kwargs):
        super(Dataset, self).__init__(data_root, human, ann_file, split)

        self.human_part = []
        self.human_part.append('body')

        # obtain the bbox for each part in canonical space, i.e., body, face, hand, bodyface, bodyhand
        if cfg.new_hand:
            self.handl_data_root = osp.join(self.data_root, 'hand/handl_new')
        else:
            self.handl_data_root = osp.join(self.data_root, 'hand/handl')
        self.handl_lbs_root = os.path.join(self.handl_data_root, 'lbs')
        self.handr_data_root = osp.join(self.data_root, 'hand/handr')
        self.handr_lbs_root = os.path.join(self.handr_data_root, 'lbs')
        self.face_root = osp.join(self.data_root, 'face')
        self.face_tvertices = np.load(osp.join(self.face_root, 'tvertices_face.npy'))

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
        # # deal with hand bbox
        # tbounds_handl[0, 0] += 0.03
        # tbounds_handr[1, 0] -= 0.03
        # # 减去left hand的space,交叉0.01m
        # tbounds_body[1, 0] = tbounds_handl[0, 0] + 0.02
        # # 减去right hand的space,交叉0.01m
        # tbounds_body[0, 0] = tbounds_handr[1, 0] - 0.02
        # # 减去face的space,交叉0.03m
        # tbounds_body[1, 1] = tbounds_face[0, 1] + 0.05
        # # dbg.showL3D([body_verts[::10], face_verts, handl_verts, tbounds_handl, tbounds_face, tbounds_body], lim=False)
        # self.tbounds_body = tbounds_body
        # self.tbounds_face = tbounds_face
        # self.tbounds_handl = tbounds_handl
        # self.tbounds_handr = tbounds_handr

    def get_sampling_points(self, bounds, N_samples):
        min_xyz = bounds[0]
        max_xyz = bounds[1]
        x_vals = np.random.uniform(0, 1, N_samples)
        y_vals = np.random.uniform(0, 1, N_samples)
        z_vals = np.random.uniform(0, 1, N_samples)
        vals = np.stack([x_vals, y_vals, z_vals], axis=1)
        pts = (max_xyz - min_xyz) * vals + min_xyz
        pts = pts.astype(np.float32)
        return pts

    def __getitem__(self, index):
        if self.human_part[index] == 'body':
            img_path = os.path.join(self.data_root, self.ims[index])
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
            tbounds = if_nerf_dutils.get_bounds(tpose)
            tbw = np.load(os.path.join(self.lbs_root, 'bigpose_bw.npy'))
            tbw = tbw.astype(np.float32)

            wpts, ppts, A, big_A, pbw, Rh, Th = self.prepare_input(i)

            pbounds = if_nerf_dutils.get_bounds(ppts)
            wbounds = if_nerf_dutils.get_bounds(wpts)

            faces_path = os.path.join(self.data_root, 'lbs/faces.npy')
            faces = np.load(faces_path)

            mesh = trimesh.Trimesh(tpose, faces, process=False)
            n_sample = 30000
            # n_sample = 100000
            wpts, ind = trimesh.sample.sample_surface_even(mesh, n_sample)
            wpts = wpts.astype(np.float32)

            sdf = np.zeros([ind.shape[0]]).astype(np.float32)
            normal = mesh.face_normals[ind].astype(np.float32)

            tpose = self.get_sampling_points(tbounds, n_sample // 10)
            tpose = tpose.astype(np.float32)

            ret = {
                'wpts': wpts,
                'sdf': sdf,
                'normal': normal,
                'tpose': tpose,
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

            # transformation
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            meta = {'R': R, 'Th': Th}
            ret.update(meta)

            latent_index = index // self.num_cams
            meta = {
                'latent_index': latent_index,
                'frame_index': frame_index,
                'part_type': 0
            }
            ret.update(meta)

        elif self.human_part[index] == 'face':

            pass

        elif self.human_part[index] == 'hand':

            img_path = os.path.join(self.data_root, self.ims[index])
            if self.human in ['CoreView_313', 'CoreView_315']:
                i = int(os.path.basename(img_path).split('_')[4])
                frame_index = i - 1
            elif 'Occlusion' in self.human:
                i = os.path.basename(img_path).split('.')[0]
                frame_index = i
            else:
                i = int(os.path.basename(img_path)[:-4])
                frame_index = i

            # read v_shaped
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
            tpose = np.load(vertices_path).astype(np.float32)
            tbounds = if_nerf_dutils.get_bounds(tpose, 'hand')
            tbw = np.load(os.path.join(self.lbs_root, 'tbw.npy'))
            tbw = tbw.astype(np.float32)

            # wpts, ppts, A, big_A, pbw, Rh, Th = self.prepare_input(i)

            # pbounds = if_nerf_dutils.get_bounds(ppts)
            # wbounds = if_nerf_dutils.get_bounds(wpts)

            faces_path = os.path.join(self.data_root, 'lbs/faces.npy')
            faces = np.load(faces_path)

            mesh = trimesh.Trimesh(tpose, faces, process=False)
            n_sample = 100000
            # from lib.utils.debugger import dbg
            # dbg.showL3D([tpose])
            wpts, ind = trimesh.sample.sample_surface_even(mesh, n_sample)
            wpts = wpts.astype(np.float32)

            sdf = np.zeros([n_sample]).astype(np.float32)
            normal = mesh.face_normals[ind].astype(np.float32)

            tpose = self.get_sampling_points(tbounds, n_sample // 10)
            tpose = tpose.astype(np.float32)

            ret = {
                'wpts': wpts,
                'sdf': sdf,
                'normal': normal,
                'tpose': tpose,
            }

            # blend weight
            meta = {
                'tbounds': tbounds
            }
            ret.update(meta)

            # transformation
            latent_index = index // self.num_cams
            meta = {
                'latent_index': latent_index
            }
            ret.update(meta)
        # meta = {'part': self.human_part[index]}
        # ret.update({'meta': meta})
        # ret.update({'tbounds_body': self.tbounds_body,
        #             'tbounds_face': self.tbounds_face,
        #             'tbounds_handl': self.tbounds_handl,
        #             'tbounds_handr': self.tbounds_handr})
        return ret

    def __len__(self):
        return len(self.human_part)
