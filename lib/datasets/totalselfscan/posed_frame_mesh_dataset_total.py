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
import openmesh


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split, **kwargs):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        test_view = [0]
        view = cfg.training_view if split == 'train' else test_view

        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        self.ims = np.array([
            np.array(ims_data['ims'])[0]
            for ims_data in annots['ims'][i:i + cfg.num_train_frame *
                                          i_intv][::i_intv]
        ]).ravel()

        if cfg.vis_tpose_mesh:
            self.ims = self.ims[:1]

        self.Ks = np.array(self.cams['K'])[cfg.training_view].astype(
            np.float32)
        self.Rs = np.array(self.cams['R'])[cfg.training_view].astype(
            np.float32)
        self.Ts = np.array(self.cams['T'])[cfg.training_view].astype(
            np.float32) / 1000.
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(
            np.float32)

        self.num_cams = 1

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))

        # obtain the bbox for each part in canonical space, i.e., body, face, hand, bodyface, bodyhand
        if cfg.new_hand:
            self.handl_data_root = osp.join(self.data_root, 'hand/handl_new')
            self.handr_data_root = osp.join(self.data_root, 'hand/handr_new')
        else:
            self.handl_data_root = osp.join(self.data_root, 'hand/handl')
            self.handr_data_root = osp.join(self.data_root, 'hand/handr')
        self.handl_lbs_root = os.path.join(self.handl_data_root, 'lbs')
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
        # tbounds_handl[0, 0] += 0.05
        # tbounds_handr[1, 0] -= 0.05
        # # deal with face bbox
        # tbounds_face[0, 1] += 0.1
        # deal with hand bbox
        tbounds_handl[0, 0] += cfg.data.hand_bound_reduce
        tbounds_handr[1, 0] -= cfg.data.hand_bound_reduce
        # deal with face bbox
        tbounds_face[0, 1] += cfg.data.face_bound_reduce
        # tbounds_face[0, 2] -= 0.05
        # tbounds_face[1, 2] += 0.05
        # 减去left hand的space,交叉0.01m
        tbounds_body[1, 0] = tbounds_handl[0, 0] + 0.02
        # 减去right hand的space,交叉0.01m
        tbounds_body[0, 0] = tbounds_handr[1, 0] - 0.02
        # 减去face的space,交叉0.03m
        tbounds_body[1, 1] = tbounds_face[0, 1] + 0.02
        # dbg.showL3D([body_verts[::10], face_verts, handl_verts, tbounds_handl, tbounds_face, tbounds_body], lim=False)
        self.tbounds_body = tbounds_body
        self.tbounds_face = tbounds_face
        self.tbounds_handl = tbounds_handl
        self.tbounds_handr = tbounds_handr

        if cfg.debug.transform_face:
            face_coord_transform_file = osp.join('data/animation', cfg.init_face, 'transform.npy')
            if osp.exists(face_coord_transform_file):
                face_coord_transform = np.load(face_coord_transform_file, allow_pickle=True).item()
                R_face2canonical = face_coord_transform['R_f'][0]
                T_face2canonical = face_coord_transform['T_f']
                self.R_face2canonical = R_face2canonical
                self.T_face2canonical = T_face2canonical

        if cfg.vis_posed_mesh:
            import open3d as o3d
            import trimesh
            # extract each part for evaluation
            bigpose_gt = o3d.io.read_triangle_mesh(osp.join(self.data_root, 'object2', 'body_000001.obj'))
            vertices = np.array(bigpose_gt.vertices)
            vertices = vertices - vertices.mean(axis=0)

            faces = np.array(bigpose_gt.triangles)
            bigpose_gt_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            smplh_face = np.load(osp.join(self.data_root, 'lbs/faces.npy'))
            bigpose_smplh_mesh = trimesh.Trimesh(vertices=body_verts, faces=smplh_face, process=False)

            # transform, _ = trimesh.registration.mesh_other(bigpose_gt_mesh, bigpose_smplh_mesh)
            transform= trimesh.registration.icp(bigpose_gt_mesh.vertices, bigpose_smplh_mesh.vertices)
            bigpose_gt_mesh.apply_transform(transform[0])
            # obtain part vertex id, face id
            # adjust the gt mesh bbox
            gt_tbounds_face = self.tbounds_face.copy()
            gt_tbounds_face[0, 1] -= cfg.eval_mesh.face_bound_add
            gt_tbounds_handl = self.tbounds_handl.copy()
            # gt_tbounds_handl[0, 0] -= 0.03
            gt_tbounds_handl[0, 1:] -= cfg.eval_mesh.hand_bound_add
            gt_tbounds_handl[1, 1:] += cfg.eval_mesh.hand_bound_add
            gt_tbounds_handr = self.tbounds_handr.copy()
            # gt_tbounds_handr[1, 0] += 0.03
            gt_tbounds_handr[0, 1:] -= cfg.eval_mesh.hand_bound_add
            gt_tbounds_handr[1, 1:] += cfg.eval_mesh.hand_bound_add
            gt_mesh_face_vert_idx = np.asarray(self.get_inside(bigpose_gt_mesh.vertices, gt_tbounds_face))
            gt_mesh_handl_vert_idx = np.asarray(self.get_inside(bigpose_gt_mesh.vertices, gt_tbounds_handl))
            gt_mesh_handr_vert_idx = np.asarray(self.get_inside(bigpose_gt_mesh.vertices, gt_tbounds_handr))
            self.gt_mesh_part_vert_idx = [gt_mesh_face_vert_idx, gt_mesh_handl_vert_idx, gt_mesh_handr_vert_idx]
            predict_mesh = o3d.io.read_triangle_mesh(osp.join('data/animation', cfg.exp_name, 'tpose_mesh.ply'))
            expand_tbounds_face = self.tbounds_face.copy()
            expand_tbounds_face[0, 2] -= 0.05
            expand_tbounds_face[1, 2] += 0.05
            expand_tbounds_handl = self.tbounds_handl.copy()
            expand_tbounds_handl[0, 1:] -= 0.02
            expand_tbounds_handl[1, 1:] += 0.02
            expand_tbounds_handr = self.tbounds_handr.copy()
            expand_tbounds_handr[0, 1:] -= 0.02
            expand_tbounds_handr[1, 1:] += 0.02
            predict_mesh_face_vert_idx = np.asarray(self.get_inside(np.asarray(predict_mesh.vertices), expand_tbounds_face))
            predict_mesh_handl_vert_idx = np.asarray(self.get_inside(np.asarray(predict_mesh.vertices), expand_tbounds_handl))
            predict_mesh_handr_vert_idx = np.asarray(self.get_inside(np.asarray(predict_mesh.vertices), expand_tbounds_handr))
            self.predict_mesh_part_vert_idx = [predict_mesh_face_vert_idx, predict_mesh_handl_vert_idx, predict_mesh_handr_vert_idx]
            # from lib.utils.debugger import dbg
            # dbg.showL3D([bigpose_gt_mesh.vertices[::10], np.asarray(predict_mesh.vertices)[::10]])
            # dbg.showL3D([bigpose_gt_mesh.vertices[gt_mesh_handr_vert_idx]])
            # dbg.showL3D([bigpose_gt_mesh.vertices[gt_mesh_face_vert_idx]])
            # import ipdb; ipdb.set_trace(context=11)



            # gt_mesh_face_faces_idx = gt_mesh_face_vert_idx[bigpose_gt_mesh.faces].sum(axis=1) == 3
            # gt_mesh_handl_faces_idx = gt_mesh_handl_vert_idx[bigpose_gt_mesh.faces].sum(axis=1) == 3
            # gt_mesh_handr_faces_idx = gt_mesh_handr_vert_idx[bigpose_gt_mesh.faces].sum(axis=1) == 3


    def get_inside(self, pts, bound):
        inside = pts > bound[:1]
        inside = inside * (pts < bound[1:])
        inside = np.sum(inside, axis=1) == 3

        return inside


    def prepare_input(self, i):
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

        return wxyz, pxyz, A, pbw, Rh, Th, big_A

    def __getitem__(self, index):
        latent_index = index
        img_path = os.path.join(self.data_root, self.ims[index])
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = os.path.basename(img_path)[:-4]
            frame_index = i
        if cfg.hand:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
            tpose = np.load(vertices_path).astype(np.float32)
            # import ipdb; ipdb.set_trace(context=11)
            # from lib.utils.debugger import dbg
            # dbg.showL3D([tpose])
            tbounds = if_nerf_dutils.get_bounds(tpose, 'hand')
        else:
            vertices_path = os.path.join(self.data_root, 'vertices','{:06d}.npy'.format(index))
            tpose = np.load(vertices_path).astype(np.float32)
            # import ipdb; ipdb.set_trace(context=11)
            # from lib.utils.debugger import dbg
            # dbg.showL3D([tpose])
            tbounds = if_nerf_dutils.get_bounds(tpose)
        body_tbounds = tbounds.copy()
        tbw = np.load(os.path.join(self.lbs_root, 'bigpose_bw.npy'))
        tbw = tbw.astype(np.float32)
        wpts, ppts, A, pbw, Rh, Th, big_A = self.prepare_input(i)
        pbounds = if_nerf_dutils.get_bounds(ppts)
        wbounds = if_nerf_dutils.get_bounds(wpts)
        if cfg.vis_face_mesh:
            # voxel_size = [0.002, 0.002, 0.002]
            # voxel_size = [0.001, 0.001, 0.001]
            voxel_size = [0.00075, 0.00075, 0.00075]
            self.face_root = osp.join(self.data_root, 'face')
            self.face_tvertices = np.load(osp.join(self.face_root, 'tvertices_face.npy'))
            tbounds = if_nerf_dutils.get_face_bounds(self.face_tvertices, delta=cfg.data.face_mesh_extract_delta)
            # deal with face bbox
            tbounds[0, 1] += cfg.data.face_bound_reduce
            # tbounds = if_nerf_dutils.get_face_bounds(self.face_tvertices, delta=0.04)

            x = np.arange(tbounds[0, 0], tbounds[1, 0] + voxel_size[0],
                          voxel_size[0])
            y = np.arange(tbounds[0, 1], tbounds[1, 1] + voxel_size[1],
                          voxel_size[1])
            z = np.arange(tbounds[0, 2], tbounds[1, 2] + voxel_size[2],
                          voxel_size[2])
        elif cfg.hand:
            voxel_size = [0.001, 0.001, 0.001]
            # voxel_size = [0.002, 0.002, 0.002]
            x = np.arange(tbounds[0, 0], tbounds[1, 0] + voxel_size[0],
                          voxel_size[0])
            y = np.arange(tbounds[0, 1], tbounds[1, 1] + voxel_size[1],
                          voxel_size[1])
            z = np.arange(tbounds[0, 2], tbounds[1, 2] + voxel_size[2],
                          voxel_size[2])
        elif cfg.vis_hand_mesh:
            if cfg.new_hand:
                if cfg.hand_type == 'handl':
                    vertices_path = os.path.join(self.data_root, 'hand/handl_new/lbs', 'bigpose_vertices.npy')
                    tvertices = np.load(vertices_path).astype(np.float32)
                else:
                    vertices_path = os.path.join(self.data_root, 'hand/handr_new/lbs', 'bigpose_vertices.npy')
                    tvertices = np.load(vertices_path).astype(np.float32)
            else:
                if cfg.hand_type == 'handl':
                    vertices_path = os.path.join(self.data_root, 'hand/handl/lbs', 'bigpose_vertices.npy')
                    tvertices = np.load(vertices_path).astype(np.float32)
                    tbw = np.load(os.path.join(self.handl_lbs_root, 'bigpose_bw.npy'))
                    tbw = tbw.astype(np.float32)
                else:
                    vertices_path = os.path.join(self.data_root, 'hand/handr/lbs', 'bigpose_vertices.npy')
                    tvertices = np.load(vertices_path).astype(np.float32)
                    tbw = np.load(os.path.join(self.handr_lbs_root, 'bigpose_bw.npy'))
                    tbw = tbw.astype(np.float32)
                # from lib.utils.debugger import dbg
                # dbg.showL3D([tpose, tvertices])
            # import ipdb; ipdb.set_trace(context=11)
            tbounds = if_nerf_dutils.get_bounds(tvertices, 'hand')
            voxel_size = [0.001, 0.001, 0.001]
            voxel_size = [0.0005, 0.0005, 0.0005]

            # voxel_size = [0.002, 0.002, 0.002]
            x = np.arange(tbounds[0, 0], tbounds[1, 0] + voxel_size[0],
                          voxel_size[0])
            y = np.arange(tbounds[0, 1], tbounds[1, 1] + voxel_size[1],
                          voxel_size[1])
            z = np.arange(tbounds[0, 2], tbounds[1, 2] + voxel_size[2],
                          voxel_size[2])
        else:
            voxel_size = cfg.voxel_size
            # voxel_size = [0.008, 0.008, 0.008]

            x = np.arange(tbounds[0, 0], tbounds[1, 0] + voxel_size[0],
                          voxel_size[0])
            y = np.arange(tbounds[0, 1], tbounds[1, 1] + voxel_size[1],
                          voxel_size[1])
            z = np.arange(tbounds[0, 2], tbounds[1, 2] + voxel_size[2],
                          voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        ret = {'pts': pts}
        meta = {
            'A': A,
            'big_A': big_A,
            'pbw': pbw,
            'tbw': tbw,
            'tpose': tpose,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds,
            'body_tbounds': body_tbounds
        }
        ret.update(meta)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': int(frame_index),
            'voxel_size': np.array(voxel_size)
        }
        ret.update(meta)
        ret.update({'tbounds_body': self.tbounds_body,
                    'tbounds_face': self.tbounds_face,
                    'tbounds_handl': self.tbounds_handl,
                    'tbounds_handr': self.tbounds_handr})
        ret.update({'part_type': cfg.part_type})
        ret.update({'meta': {'data_root': self.data_root}})
        if hasattr(self, 'gt_mesh_part_vert_idx'):
            ret['meta'].update({'gt_mesh_part_vert_idx': self.gt_mesh_part_vert_idx})
        if hasattr(self, 'predict_mesh_part_vert_idx'):
            ret['meta'].update({'predict_mesh_part_vert_idx': self.predict_mesh_part_vert_idx})
        if cfg.debug.transform_face:
            ret.update({'T_canonical2face': - self.T_face2canonical@self.R_face2canonical,
                        'R_canonical2face': self.R_face2canonical.T})


        return ret

    def __len__(self):
        return len(self.ims)

