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
        self.face_vis_statics = {0: {}, 1: {}, 2:{}, 3: {}}
        # for part in self.face_vis_statics.keys():
        #     self.face_vis_statics[part]

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
                vertex = self.tvertex[None]
            self.pvertex = vertex[0]

            # perform the rasterization
            ppose = struct.Meshes(verts=vertex, faces=self.tface[None])
            fragments = rasterizer(ppose)
            face_idx_map = fragments.pix_to_face[0, ..., 0]
            mask = face_idx_map > 0
            mask = batch['msk'][0] * mask
            row_idx, col_idx = torch.where(mask!=0)
            row_idx, col_idx = row_idx.tolist(), col_idx.tolist()
            crop_bbox = batch['crop_bbox'][0, 0].cpu() / cfg.ratio
            part = batch['part_type'].item()
            latent_index = batch['latent_index'].item()
            for i in range(len(row_idx)):
                face = face_idx_map[row_idx[i], col_idx[i]].item()
                if batch['face_label'][0, face].item() != part:
                    continue
                if face not in self.face_vis_statics[part].keys():
                    self.face_vis_statics[part][face] = {}
                    self.face_vis_statics[part][face]['num'] = 0
                    self.face_vis_statics[part][face]['frame_coord'] = {}
                self.face_vis_statics[part][face]['num'] += 1
                if latent_index not in self.face_vis_statics[part][face]['frame_coord'].keys():
                    self.face_vis_statics[part][face]['frame_coord'][latent_index] = []
                self.face_vis_statics[part][face]['frame_coord'][latent_index].append([int(row_idx[i]/cfg.ratio+crop_bbox[1]),
                                                                                       int(col_idx[i]/cfg.ratio+crop_bbox[0])])
        ret = {}
        ret['face_vis_statics'] = self.face_vis_statics
        ret['n_face'] = self.tface.shape[0]
        return ret
