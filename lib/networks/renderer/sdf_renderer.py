import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *


class Renderer:
    def __init__(self, net):
        self.net = net

    def get_wsampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        # sample mid-point
        # dists = z_vals[..., 1:] - z_vals[..., :-1]
        # dists = torch.cat([dists, dists[..., -1:]], dim=2)
        # mid_z_vals = z_vals + dists * 0.5
        # pts = ray_o[:, :, None] + ray_d[:, :, None] * mid_z_vals[..., None]
        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

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

    def get_pixel_value(self, ray_o, ray_d, near, far, occ, batch):
        n_batch = ray_o.shape[0]

        # sampling points
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        if 'tpose_human' in dir(self.net):
            sdf_network = self.net.tpose_human
        else:
            sdf_network = self.net
        raw_decoder = lambda wpts_val, viewdir_val, z_vals_val: sdf_network(
            wpts_val, viewdir_val, z_vals_val, batch)

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].view(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        gradients = ret['gradients'].view(n_batch, n_pixel * n_sample, 3)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        ret.update({
            'rgb_map': rgb_map,
            'gradients': gradients,
            'acc_map': acc_map,
            'depth_map': depth_map
        })

        # separate points into inside and outside points
        sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)
        free_sdf = sdf[occ == 0].view(-1)[None]
        ret.update({'free_sdf': free_sdf})

        occ_sdf = sdf[occ == 1].min(dim=1)[0]
        occ_sdf = occ_sdf[None]
        ret.update({'occ_sdf': occ_sdf})

        return ret

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        occ = batch['occupancy']
        sh = ray_o.shape

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            occ_chunk = occ[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               occ_chunk, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
