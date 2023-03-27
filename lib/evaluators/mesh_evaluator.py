import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import math
import os
from os.path import join
from PIL import Image
from lib.config import cfg
import open3d as o3d
from termcolor import colored
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils


class Evaluator:
    def __init__(self) -> None:
        self.p2ss = []
        self.chamfers = []
        self.part_p2ss = [[], [], []]
        self.part_chamfers = [[], [], []]
        self.parts = ['face', 'handl', 'handr']
        self.mesh_eval = MeshEvaluator()
    def get_inside(self, pts, bound):
        inside = pts > bound[:1]
        inside = inside * (pts < bound[1:])
        inside = np.sum(inside, axis=1) == 3

        return inside
    def evaluate(self, output, batch):
        human = cfg.test_dataset.human
        vertices = output['posed_vertex']
        if 'rp' in human:
            new_vertices = np.zeros_like(vertices)
            new_vertices[:, 0] = vertices[:, 0]
            new_vertices[:, 1] = vertices[:, 2]
            new_vertices[:, 2] = -vertices[:, 1]
            vertices = new_vertices
        # move to zero
        head_top_point = np.argmin(vertices[:, 1])
        vertices = vertices - vertices[head_top_point:head_top_point+1]
        src_mesh = trimesh.Trimesh(vertices, output['triangle'], process=False)
        data_root = batch['meta']['data_root'][0]
        # data_root = cfg.test_dataset.data_root
        frame_index = batch['frame_index'].item() + 2
        tgt_mesh_path = os.path.join(data_root,
                                     'object2/body_{:06d}.obj'.format(frame_index))
        tgt_mesh = o3d.io.read_triangle_mesh(tgt_mesh_path)
        # bigpose_gt = o3d.io.read_triangle_mesh(os.path.join(data_root, 'untitled.obj'))


        # import ipdb; ipdb.set_trace(context=11)
        # from lib.utils.debugger import dbg
        # dbg.showL3D([vertices[::10], np.array(tgt_mesh.vertices)[::10]])
        gt_head_top_point = np.argmin(np.array(tgt_mesh.vertices)[:, 1])
        tgt_mesh = trimesh.Trimesh(np.array(tgt_mesh.vertices)-np.array(tgt_mesh.vertices)[gt_head_top_point:gt_head_top_point+1],
                                   tgt_mesh.triangles,
                                   process=False)
        # import ipdb; ipdb.set_trace(context=11)
        # from lib.utils.debugger import dbg
        # dbg.showL3D([tgt_mesh.vertices[::20], src_mesh.vertices[::20]])
        self.mesh_eval.set_src_mesh(src_mesh)
        self.mesh_eval.set_tgt_mesh(tgt_mesh)
        self.mesh_eval.apply_pts_registration()
        # self.mesh_eval.apply_registration()
        register_mesh_path = cfg.result_dir
        os.makedirs(register_mesh_path, exist_ok=True)
        self.mesh_eval.src_mesh.export(os.path.join(register_mesh_path, 'predicted_{}.obj'.format(frame_index)))
        self.mesh_eval.tgt_mesh.export(os.path.join(register_mesh_path, 'gt_{}.obj'.format(frame_index)))
        chamfer = self.mesh_eval.get_chamfer_dist()
        p2s = self.mesh_eval.get_surface_dist()
        self.chamfers.append(chamfer)
        self.p2ss.append(p2s)
        if cfg.evaluation.neuralbody or cfg.evaluation.aninerf:
            face_verts = self.mesh_eval.tgt_mesh.vertices[batch['meta']['gt_mesh_part_vert_idx'][0][0]]
            expand_tbounds_face = if_nerf_dutils.get_bounds(face_verts, 'face')
            expand_tbounds_face[0, 2] -= 0.05
            expand_tbounds_face[0, 0] -= 0.05
            expand_tbounds_face[1, 2] += 0.05
            expand_tbounds_face[1, 0] += 0.05
            handl_verts = self.mesh_eval.tgt_mesh.vertices[batch['meta']['gt_mesh_part_vert_idx'][1][0]]
            expand_tbounds_handl = if_nerf_dutils.get_bounds(handl_verts, 'hand')
            expand_tbounds_handl[0, 1:] -= 0.05
            expand_tbounds_handl[1, 1:] += 0.05
            handr_verts = self.mesh_eval.tgt_mesh.vertices[batch['meta']['gt_mesh_part_vert_idx'][2][0]]
            expand_tbounds_handr = if_nerf_dutils.get_bounds(handr_verts, 'hand')
            expand_tbounds_handr[0, 1:] -= 0.05
            expand_tbounds_handr[1, 1:] += 0.05
            predict_mesh_face_vert_idx = np.asarray(self.get_inside(np.asarray(self.mesh_eval.src_mesh.vertices), expand_tbounds_face))
            predict_mesh_handl_vert_idx = np.asarray(self.get_inside(np.asarray(self.mesh_eval.src_mesh.vertices), expand_tbounds_handl))
            predict_mesh_handr_vert_idx = np.asarray(self.get_inside(np.asarray(self.mesh_eval.src_mesh.vertices), expand_tbounds_handr))
            predict_mesh_part_vert_idx = [predict_mesh_face_vert_idx, predict_mesh_handl_vert_idx, predict_mesh_handr_vert_idx]
            predict_part_meshs = gen_part_mesh(src_mesh, predict_mesh_part_vert_idx)
        else:
            predict_part_meshs = gen_part_mesh(src_mesh, batch['meta']['predict_mesh_part_vert_idx'])
        gt_part_meshs = gen_part_mesh(tgt_mesh, batch['meta']['gt_mesh_part_vert_idx'])
        for part in range(len(gt_part_meshs)):
            part_src_mesh = predict_part_meshs[part]
            part_tgt_mesh = gt_part_meshs[part]
            self.mesh_eval.set_src_mesh(part_src_mesh)
            self.mesh_eval.set_tgt_mesh(part_tgt_mesh)
            # self.mesh_eval.apply_registration()
            self.mesh_eval.apply_pts_registration()
            self.mesh_eval.src_mesh.export(os.path.join(register_mesh_path, 'predicted_{}_{}.obj'.format(frame_index, self.parts[part])))
            self.mesh_eval.tgt_mesh.export(os.path.join(register_mesh_path, 'gt_{}_{}.obj'.format(frame_index, self.parts[part])))
            chamfer = []
            p2s = []
            for i in range(5):
                chamfer.append(self.mesh_eval.get_chamfer_dist())
                p2s.append(self.mesh_eval.get_surface_dist())
            self.part_chamfers[part].append(sum(chamfer)/len(chamfer))
            self.part_p2ss[part].append(sum(p2s)/len(p2s))

    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))
        result_path = os.path.join(cfg.result_dir, 'mesh_metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        metrics = {'p2s': self.p2ss, 'chamfer': self.chamfers}
        np.save(result_path, metrics)

        print('p2s: {}'.format(np.mean(self.p2ss)))
        print('chamfer: {}'.format(np.mean(self.chamfers)))
        for i, part in enumerate(self.parts):
            print('{} p2s: {}'.format(part, np.mean(self.part_p2ss[i])))
            print('{} chamfer: {}'.format(part, np.mean(self.part_chamfers[i])))

        self.p2ss = []
        self.chamfers = []

def gen_part_mesh(posed_mesh: trimesh.Trimesh, part_vert_idx: list)->list:
    all_part_mesh = []
    for part_i_vert_idx in part_vert_idx:
        part_i_vert_idx = part_i_vert_idx.squeeze()
        part_vert = posed_mesh.vertices[part_i_vert_idx]
        part_faces_idx = part_i_vert_idx[posed_mesh.faces].sum(axis=1) == 3
        part_faces = posed_mesh.faces[part_faces_idx]
        new_idx = np.zeros(posed_mesh.vertices.shape[0])
        new_idx[part_i_vert_idx] = np.arange(part_vert.shape[0])
        new_part_faces = new_idx[part_faces]
        part_mesh = trimesh.Trimesh(vertices=part_vert, faces=new_part_faces, process=False)
        all_part_mesh.append(part_mesh)
    return all_part_mesh

class MeshEvaluator:
    """
    From https://github.com/facebookresearch/pifuhd/blob/master/lib/evaluator.py
    """
    _normal_render = None

    def __init__(self, scale_factor=1.0, offset=0):
        self.scale_factor = scale_factor
        self.offset = offset
        pass

    def set_mesh(self, src_path, tgt_path):
        self.src_mesh = trimesh.load(src_path)
        self.tgt_mesh = trimesh.load(tgt_path)

    def apply_registration(self):
        transform, _ = trimesh.registration.mesh_other(self.src_mesh,
                                                       self.tgt_mesh)
        self.src_mesh.apply_transform(transform)

    def apply_pts_registration(self):
        transform, _, _ = trimesh.registration.icp(self.src_mesh.vertices,
                                                       self.tgt_mesh.vertices, scale=False)
        self.src_mesh.apply_transform(transform)

    def set_src_mesh(self, mesh):
        self.src_mesh = mesh

    def set_tgt_mesh(self, mesh):
        self.tgt_mesh = mesh

    def get_chamfer_dist(self, num_samples=1000):
        # breakpoint()
        # Chamfer
        src_surf_pts, _ = trimesh.sample.sample_surface(
            self.src_mesh, num_samples)
        # self.src_mesh.show()
        tgt_surf_pts, _ = trimesh.sample.sample_surface(
            self.tgt_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(
            self.tgt_mesh, src_surf_pts)
        _, tgt_src_dist, _ = trimesh.proximity.closest_point(
            self.src_mesh, tgt_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0
        tgt_src_dist[np.isnan(tgt_src_dist)] = 0

        src_tgt_dist = src_tgt_dist.mean()
        tgt_src_dist = tgt_src_dist.mean()

        chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

        return chamfer_dist

    def get_surface_dist(self, num_samples=10000):
        # P2S
        src_surf_pts, _ = trimesh.sample.sample_surface(
            self.src_mesh, num_samples)

        _, src_tgt_dist, _ = trimesh.proximity.closest_point(
            self.tgt_mesh, src_surf_pts)

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0

        src_tgt_dist = src_tgt_dist.mean()

        return src_tgt_dist
