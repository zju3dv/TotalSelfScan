import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
from scipy import stats
from skimage import measure
import trimesh
from lib.utils.blend_utils import *
from os.path import join


class Renderer:
    def __init__(self, net):
        self.net = net

    def batchify_rays(self, wpts, sdf_decoder, net, chunk, batch):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            # resd = net.calculate_residual_deformation(wpts[i:i + chunk][None],
            #                                           batch['latent_index'])
            # wpts[i:i + chunk] = wpts[i:i + chunk] + resd[0]
            ret = sdf_decoder(wpts[i:i + chunk])[:, :1]
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret

    def batchify_blend_weights(self, pts, bw_input, chunk=1024 * 32):
        all_ret = []
        for i in range(0, pts.shape[1], chunk):
            ret = self.net.calculate_bigpose_smpl_bw(pts[:, i:i + chunk],
                                                     bw_input)
            all_ret.append(ret)
        all_ret = torch.cat(all_ret, 2)
        return all_ret

    def batchify_normal_sdf(self, pts, batch, chunk=1024 * 32):
        all_normal = []
        all_sdf = []
        for i in range(0, pts.shape[1], chunk):
            normal, sdf = self.net.gradient_of_deformed_sdf(
                pts[:, i:i + chunk], batch)
            all_normal.append(normal.detach().cpu().numpy())
            all_sdf.append(sdf.detach().cpu().numpy())
        all_normal = np.concatenate(all_normal, axis=1)
        all_sdf = np.concatenate(all_sdf, axis=1)
        return all_normal, all_sdf

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape

        pts = pts.view(sh[0], -1, 3)

        # sampling points for blend weight training
        if 'tbw' in batch and not cfg.vis_face_mesh and not cfg.vis_hand_mesh:
            bw = pts_sample_blend_weights(pts, batch['tbw'], batch['tbounds'])
            tnorm = bw[:, -1]
            inside = tnorm < cfg.norm_th
        else:
            inside = torch.ones(pts.shape[:2]).bool()

        pts = pts[inside]
        if 'sdf_network' in dir(self.net):
            sdf_decoder = lambda x: self.net.sdf_network(x, batch)
        elif cfg.vis_union:
            tpose_human = self.net.tpose_human
            sdf_decoder = lambda x: tpose_human.final_sdf_network(x, batch, multi=cfg.vis_mesh_multi)
        else:
            tpose_human = self.net.tpose_human
            sdf_decoder = lambda x: tpose_human.sdf_network(x, batch, multi=cfg.vis_mesh_multi)
        with torch.no_grad():
            sdf = self.batchify_rays(pts, sdf_decoder, self.net, 2048 * 16, batch)
        # sdf = self.batchify_rays(pts, sdf_decoder, self.net, 2048 * 64, batch)

        inside = inside.detach().cpu().numpy()
        full_sdf = 10 * np.ones(inside.shape)
        full_sdf[inside] = sdf[:, 0]
        sdf = -full_sdf

        cube = sdf.reshape(*sh[1:-1])
        cube = np.pad(cube, 10, mode='constant', constant_values=-10)
        # vertices, triangles, normals, values = measure.marching_cubes(cube, cfg.mesh_th)
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        mesh = trimesh.Trimesh(vertices, triangles)
        vertices = (vertices - 10) * batch['voxel_size'][0, 0].item()
        # vertices = (vertices - 10) * cfg.voxel_size[0]
        vertices = vertices + batch['tbounds'][0, 0].detach().cpu().numpy()
        # import ipdb; ipdb.set_trace(context=11)
        # from lib.utils.debugger import dbg
        # dbg.showL3D([vertices])
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        # triangles = triangles[labels == 0]
        # select the labels with most numbers
        select_label = stats.mode(labels)[0][0]
        triangles = triangles[labels == select_label]
        import open3d as o3d
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
        mesh_o3d.remove_unreferenced_vertices()
        vertices = np.array(mesh_o3d.vertices)
        triangles = np.array(mesh_o3d.triangles)
        # if cfg.hand or True:
        if False:
            ret = {
                'vertex': vertices,
                'triangle': triangles,
                # 'rgb': rgb,
            }
            return ret

        # transform vertices to the world space
        pts = torch.from_numpy(vertices).to(pts)[None]
        if cfg.vis_face_mesh:
            tbw = pts_sample_blend_weights(pts, batch['tbw'], batch['body_tbounds'])
        else:
            tbw = pts_sample_blend_weights(pts, batch['tbw'], batch['tbounds'])
        tbw = tbw[:, :-1]

        if 'gradient_of_deformed_sdf' in dir(self.net) and False:
            normal, sdf = self.batchify_normal_sdf(pts, batch)
            resd = -normal * sdf
            resd = torch.from_numpy(resd).to(pts)
            deformed_pts = pts + resd
        else:
            deformed_pts = pts

        tpose_pts = pose_points_to_tpose_points(deformed_pts, tbw,
                                                batch['big_A'])
        pose_pts = tpose_points_to_pose_points(tpose_pts, tbw, batch['A'])
        pose_pts = pose_points_to_world_points(pose_pts, batch['R'],
                                               batch['Th'])
        posed_vertices = pose_pts[0].detach().cpu().numpy()

        ret = {
            'vertex': vertices,
            'posed_vertex': posed_vertices,
            'triangle': triangles,
            # 'rgb': rgb,
        }
        if cfg.evaluation.neuralbody:
            frame_index = batch['frame_index'].item()
            sub = cfg.exp_name.split('_')[1]
            result_path = join('/nas/datasets/total_capture/paper/synthsis/baseline/neuralbody/mesh', sub, 'mesh')
            mesh_name = join(result_path, '{:04d}.ply'.format(frame_index))
            from pytorch3d.io import IO
            pred_mesh = IO().load_mesh(mesh_name)
            ret['posed_vertex'] = pred_mesh.verts_list()[0].numpy()
            ret['triangle'] = pred_mesh.faces_list()[0].numpy()

        if cfg.evaluation.aninerf:
            frame_index = batch['frame_index'].item()
            sub = cfg.exp_name.split('_')[1]
            result_path = join('/nas/datasets/total_capture/paper/synthsis/baseline/aninerf/mesh', 'aninerf_syn_{}'.format(sub), 'posed_mesh')
            mesh_name = join(result_path, '{:04d}.ply'.format(frame_index))
            from pytorch3d.io import IO
            pred_mesh = IO().load_mesh(mesh_name)
            ret['posed_vertex'] = pred_mesh.verts_list()[0].numpy()
            ret['triangle'] = pred_mesh.faces_list()[0].numpy()

        if 'tbw' in batch:
            pts = torch.tensor(vertices[None]).to(pts)
            pts = pts.view(sh[0], -1, 3)
            bw_input = {
                'tbw': batch['tbw'],
                'tbounds': batch['tbounds'],
                'latent_index': batch['latent_index']
            }
            # blend_weights = self.sample_bw_knn(pts, batch)
            blend_weights = self.sample_bw_closest_face(pts, batch)

            # blend_weights = self.batchify_blend_weights(pts, bw_input)
            blend_weights = blend_weights[0].detach().cpu().numpy()
            ret.update({'blend_weight': blend_weights})

        return ret

    def sample_bw_knn(self, pts, batch):
        from lib.networks.total.network_optimizePose import sampleClosestPointsWithKNN
        from lib.utils.base_utils import read_pickle
        smplh = read_pickle('data/smplx/smplh/SMPLH_MALE.pkl')
        smplh_bw = smplh['weights'].T.astype(np.float32)
        smplh_bw = torch.from_numpy(smplh_bw).to(pts)[None]
        pvertex_i = batch['tpose']
        pts_pbw, dists = sampleClosestPointsWithKNN(pts, pvertex_i, smplh_bw.permute([0, 2, 1]))
        blend_weights = torch.cat([pts_pbw.permute(0, 2, 1), dists[None, None]], dim=1)

        return blend_weights

    def sample_bw_closest_face(self, pts, batch):
        from psbody.mesh import Mesh
        from lib.utils.base_utils import read_pickle
        smplh = read_pickle('data/smplx/smplh/SMPLH_MALE.pkl')
        smplh_bw = smplh['weights'].astype(np.float32)
        pts = pts[0].cpu().numpy()
        vertices = batch['tpose'][0].cpu().numpy()
        faces = smplh['f']
        mesh = Mesh(vertices, faces)
        closest_faces, closest_points = mesh.closest_faces_and_points(pts)
        vert_ids, bary_coords = mesh.barycentric_coordinates_for_points(closest_points, closest_faces)
        pts_bweights = barycentric_interpolation(smplh_bw[vert_ids], bary_coords)
        # calculate the distance of grid_points to mano mehs
        distance = np.linalg.norm(pts - closest_points, axis=1)
        result = np.concatenate([pts_bweights, distance[..., None]], axis=1)[None]
        result = torch.from_numpy(result).permute(0, 2, 1).to('cuda')

        return result

def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret