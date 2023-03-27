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
# import open3d as o3d
# from lib.utils.vis_utils import get_colored_pc


class Renderer:
    def __init__(self, net):
        tpose_dir = 'data/animation/{}/mesh.npy'.format(cfg.exp_name)
        tpose_mesh = np.load(tpose_dir, allow_pickle=True).item()
        self.net = net
        self.tvertex = torch.from_numpy(tpose_mesh['vertex']).cuda().float()
        triangle = tpose_mesh['triangle'].astype(np.int32)
        self.tface = torch.from_numpy(triangle).cuda().float()
        self.bw = torch.from_numpy(
            tpose_mesh['blend_weight'][:-1]).cuda()[None].float()

    def get_wsampling_points(self, ray_o, ray_d, wpts):
        # calculate the steps for each ray
        n_batch, num_rays = ray_d.shape[:2]
        z_interval = cfg.z_interval
        n_sample = cfg.N_samples // 2 * 2 + 1
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
            pixel_vertex = self.pvertex[pixel_vertex_idx, :]
            wpts = torch.sum(pixel_vertex * bary_coords[..., None], dim=2)
            wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, wpts)

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        raw_decoder = lambda wpts_val, viewdir_val, dists_val: self.net(
            wpts_val, viewdir_val, dists_val, batch)

        # compute the color and density
        wpts = wpts.reshape(n_batch, n_pixel, -1, 3)
        ret = self.get_density_color(wpts, viewdir, z_vals, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].reshape(-1, n_sample, 65)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        ret = ({
            'rgb_map': rgb_map,
            'acc_map': acc_map,
            'depth_map': depth_map,
        })

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
        R = batch['R_']
        T = batch['T']
        can_idx = batch['cam_ind']
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

            # pose the mesh
            ppose = pose_points_to_tpose_points(self.tvertex[None], self.bw,
                                                batch['big_A'])
            pvertex_i = tpose_points_to_pose_points(ppose, self.bw, batch['A'])
            vertex = pose_points_to_world_points(pvertex_i, batch['R'],
                                                 batch['Th'])
            self.pvertex = vertex[0]

            # perform the rasterization
            ppose = struct.Meshes(verts=vertex, faces=self.tface[None])
            fragments = rasterizer(ppose)
            face_idx_map = fragments.pix_to_face[0, ..., 0]
            mask = face_idx_map > 0

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
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk, batch,
                                               face_idx_chunk,
                                               bary_coords_chunk)
            ret_list.append(pixel_value)
        # from lib.utils.debugger import dbg
        # dbg.showL3D([vertex[0].cpu()])

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        mask_at_box = batch['mask_at_box']
        H, W = mask_at_box.shape[1:]
        pixel_rgb_map = ret['rgb_map']
        n_channel = pixel_rgb_map.shape[2]
        rgb_map = torch.zeros([n_batch, H, W, n_channel]).to(pixel_rgb_map)
        rgb_map[mask[None]] = pixel_rgb_map[0]
        rgb_map = rgb_map.permute(0, 3, 1, 2).contiguous()
        raw = self.net.image_renderer(rgb_map, batch)
        pred_img = raw[:, :3]
        pred_msk = raw[:, 3]

        ret.update({'pred_img': pred_img, 'pred_msk': pred_msk})

        return ret
