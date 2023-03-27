import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)
import pytorch3d.structures as struct


class Renderer:
    def __init__(self, net):
        tpose_dir = 'data/animation/{}/tpose_mesh.npy'.format(cfg.exp_name)
        tpose_mesh = np.load(tpose_dir, allow_pickle=True).item()
        self.net = net
        self.tvertex = torch.from_numpy(tpose_mesh['vertex']).cuda().float()
        triangle = tpose_mesh['triangle'].astype(np.int32)
        self.tface = torch.from_numpy(triangle).cuda().float()
        self.bw = torch.from_numpy(
            tpose_mesh['blend_weight'][:-1]).cuda()[None].float()
        self.view_dirs_statics = {0: {}, 1: {}, 2:{}, 3: {}}

    def get_wsampling_points(self, ray_o, ray_d, wpts):
        # calculate the steps for each ray
        n_batch, num_rays = ray_d.shape[:2]
        z_interval = cfg.z_interval
        n_sample = cfg.N_samples // 2 * 2 + 1
        # 这里sample的是前后z_interval距离的点，如果z_interval太大，就会采到其他的点
        z_vals = torch.linspace(-z_interval, z_interval,
                                steps=n_sample).to(ray_d)

        if cfg.perturb > 0. and self.net.training:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape).to(ray_d)
            z_vals = lower + (upper - lower) * t_rand

        pts = wpts[:, :, None] + ray_d[:, :, None] * z_vals[:, None]
        z_vals = (pts[..., 0] - ray_o[..., :1]) / ray_d[..., :1]

        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, raw_decoder):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch * n_pixel * n_sample)

        ret = raw_decoder(wpts, viewdir, dists)

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, batch, face_idx,
                        bary_coords):
        n_batch = ray_o.shape[0]
        n_pixel = face_idx.shape[1]
        # sampling points for nerf training
        with torch.no_grad():
            pixel_vertex_idx = self.tface[face_idx].long()

            # get 3d points
            pixel_vertex = self.pvertex[pixel_vertex_idx, :]
            wpts = torch.sum(pixel_vertex * bary_coords[..., None], dim=2)

            # get blend weights of 3d points
            pixel_bw = self.bw[0].transpose(0, 1)[pixel_vertex_idx]
            wpts_bw = torch.sum(pixel_bw * bary_coords[..., None], dim=2)

            wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, wpts)
            n_sample = wpts.shape[2]
            wpts_bw = wpts_bw[:, :, None].expand(-1, -1, n_sample,
                                                 -1).contiguous()
            wpts_ray_d = ray_d[:, :, None].expand(-1, -1, n_sample,
                                                  -1).contiguous()
        num_joints = wpts_bw.shape[-1]
        wpts_bw = wpts_bw.view(1, n_pixel * n_sample, num_joints).permute(0, 2, 1)
        wpts_ray_d = wpts_ray_d.view(1, n_pixel * n_sample, 3)

        pose_dirs = world_dirs_to_pose_dirs(wpts_ray_d, batch['R'])

        part_type = batch['part_type'].item()
        if part_type in [0, 2, 3]:
            # transform points from the pose space to the tpose space
            init_tdirs = pose_dirs_to_tpose_dirs_rigid(pose_dirs, wpts_bw, batch['A'])
            tpose_dirs = tpose_dirs_to_pose_dirs_rigid(init_tdirs, wpts_bw,
                                                 batch['big_A'])
        else:
            # if cfg.debug.transform_face:
            #     pose_dirs[0] = pose_dirs[0] @ batch['R_canonical2face'][0].t()
            tpose_dirs = pose_dirs

        if True:
            self.statics_viewdir_vertex(part_type, tpose_dirs, face_idx, bary_coords)
        if False:
            self.statics_viewdir_face(part_type, tpose_dirs, face_idx)

    def statics_viewdir_vertex(self, part_type, tpose_dirs, face_idx, bary_coords):
        for i in range(face_idx.shape[1]):
            face_i = face_idx[0, i].item()
            # get the vertex id
            vertex = self.tface[face_i]
            max_vertex_idx = torch.argmax(bary_coords[0, i])
            select_vertex = vertex[max_vertex_idx].item()
            if select_vertex not in self.view_dirs_statics[part_type].keys():
                self.view_dirs_statics[part_type][select_vertex] = tpose_dirs[0, i:i+1]
            else:
                self.view_dirs_statics[part_type][select_vertex] = torch.cat((self.view_dirs_statics[part_type][select_vertex],
                                                                       tpose_dirs[0, i:i+1]), dim=0)

    def statics_viewdir_face(self, part_type, tpose_dirs, face_idx):
        for i in range(face_idx.shape[1]):
            face_i = face_idx[0, i].item()
            if face_i not in self.view_dirs_statics[part_type].keys():
                self.view_dirs_statics[part_type][face_i] = tpose_dirs[0, i:i+1]
            else:
                self.view_dirs_statics[part_type][face_i] = torch.cat((self.view_dirs_statics[part_type][face_i],
                                                                       tpose_dirs[0, i:i+1]), dim=0)


    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        # volume rendering for each pixel
        n_batch = ray_o.shape[0]
        ret_list = []
        pytorch3d_K = batch['pytorch3d_K']
        R = batch['cam_R']
        T = batch['cam_T']
        with torch.no_grad():
            # set camera
            cameras = PerspectiveCameras(device=ray_o.device,
                                         K=pytorch3d_K.float(),
                                         R=R[0].T[None],
                                         T=T[0].T.float())
            height, width = batch['img'].shape[-2:]
            raster_settings = RasterizationSettings(image_size=(height, width),
                                                    blur_radius=0.0,
                                                    faces_per_pixel=1,
                                                    bin_size=None)
            rasterizer = MeshRasterizer(cameras=cameras,
                                        raster_settings=raster_settings)

            if batch['part_type'].item() in [0, 2, 3]:
                # pose the mesh
                ppose = pose_points_to_tpose_points(self.tvertex[None], self.bw,
                                                    batch['big_A'])
                if hasattr(self.net, 'getOptimizedTransformationMatrixAndDeltaPose'):
                    A, _ = self.net.getOptimizedTransformationMatrixAndDeltaPose(batch)
                    A = A[None]
                else:
                    A = batch['A']
                pvertex_i = tpose_points_to_pose_points(ppose, self.bw, A)
                vertex = pose_points_to_world_points(pvertex_i, batch['R'],
                                                     batch['Th'])
            else:
                if cfg.debug.transform_face:
                    self.tvertex_transform = self.tvertex @ batch['R_canonical2face'][0].t() + batch['T_canonical2face'][0]
                    vertex = self.tvertex_transform[None]
                else:
                    vertex = self.tvertex[None]
            self.pvertex = vertex[0]

            # perform the rasterization
            ppose = struct.Meshes(verts=vertex, faces=self.tface[None])
            fragments = rasterizer(ppose)
            face_idx_map = fragments.pix_to_face[0, ..., 0]
            mask = face_idx_map > 0
            # import matplotlib.pylab as plt;plt.figure();plt.imshow(mask.cpu());plt.show()
            # img = batch['img'][0].permute([1,2,0]).cpu()
            # import cv2
            # mask = mask * 255
            # cv2.imwrite('./mask1.jpg', mask.cpu().numpy().astype(np.uint8))

            face_idx_map = face_idx_map[mask][None]
            bary_coords_map = fragments.bary_coords[0, :, :, 0]
            bary_coords_map = bary_coords_map[mask][None]

            ray_o = ray_o[0][mask][None]
            ray_d = ray_d[0][mask][None]
            n_pixel = ray_o.shape[1]

        chunk = 2048 * 3
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            face_idx_chunk = face_idx_map[:, i:i + chunk]
            bary_coords_chunk = bary_coords_map[:, i:i + chunk]
            self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk, batch,
                                               face_idx_chunk,
                                               bary_coords_chunk)
        ret = {}
        ret['view_dirs_statics'] = self.view_dirs_statics
        ret['n_face'] = self.tface.shape[0]
        return ret
