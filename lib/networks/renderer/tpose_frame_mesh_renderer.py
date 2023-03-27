import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh
from lib.utils.blend_utils import *


class Renderer:
    def __init__(self, net):
        self.net = net

    def sdf_decoder(self, pose_pts, batch):
        """
        pose_pts: num_points, 3
        """
        pose_pts = pose_pts[None]
        tpose, _, init_bigpose, resd = self.net.pose_points_to_tpose_points(
            pose_pts, None, batch)
        tpose = tpose[0]
        sdf = self.net.tpose_human.sdf_network(tpose, batch)
        return sdf

    def batchify_rays(self, pts, sdf_decoder, chunk=1024 * 32):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = pts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = sdf_decoder(pts[i:i + chunk])[:, :1]
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret

    def render(self, batch):
        wpts = batch['pts']
        sh = wpts.shape

        wpts = wpts.view(sh[0], -1, 3)
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        # sampling points for blend weight training
        if 'pbw' in batch:
            bw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                          batch['pbounds'])
            pnorm = bw[:, 24]
            inside = pnorm < cfg.norm_th
        else:
            inside = torch.ones(pose_pts.shape[:2]).bool()

        pose_pts = pose_pts[inside]

        sdf_decoder = lambda x: self.sdf_decoder(x, batch)

        sdf = self.batchify_rays(pose_pts, sdf_decoder, 2048 * 64)

        inside = inside.detach().cpu().numpy()
        full_sdf = 10 * np.ones(inside.shape)
        full_sdf[inside] = sdf[:, 0]
        sdf = -full_sdf

        cube = sdf.reshape(*sh[1:-1])
        cube = np.pad(cube, 10, mode='constant', constant_values=-10)
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        vertices = (vertices - 10) * cfg.voxel_size[0]
        vertices = vertices + batch['wbounds'][0, 0].detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices, triangles)
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        triangles = triangles[labels == 0]
        import open3d as o3d
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
        mesh_o3d.remove_unreferenced_vertices()
        vertices = np.array(mesh_o3d.vertices)
        triangles = np.array(mesh_o3d.triangles)

        ret = {
            'posed_vertex': vertices,
            'triangle': triangles,
        }

        return ret
