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

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def sample_around_surface(self, surf_msk, surf_z_ind, z_vals, ray_o, ray_d,
                              near, far, batch):
        """
        surf_msk: n_batch, n_pixel
        surf_z_ind: n_batch, n_pixel
        z_vals: n_batch, n_pixel, n_sample
        ray_o: n_batch, n_pixel, 3
        ray_d: n_batch, n_pixel, 3
        near: n_batch, n_pixel
        far: n_batch, n_pixel
        """
        n_batch, n_pixel, n_sample = z_vals.shape
        batch_ind = torch.arange(n_batch)
        pixel_ind = torch.arange(n_pixel)

        intv = 16
        if 'iter_step' in batch:
            intv = int(intv * (0.5**(batch['iter_step'] / 5000)))
        intv = max(intv, 4)

        near_z_ind = torch.clamp(surf_z_ind - intv, min=0)
        far_z_ind = torch.clamp(surf_z_ind + intv, max=n_sample - 1)

        near_z = z_vals[batch_ind, pixel_ind, near_z_ind]
        far_z = z_vals[batch_ind, pixel_ind, far_z_ind]

        near[surf_msk] = near_z[surf_msk]
        far[surf_msk] = far_z[surf_msk]

        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)

        return wpts, z_vals

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

    def get_intersection_point(self, intersection_mask, occ, wpts, z_vals,
                               ray_o, ray_d, ret, batch):
        """
        intersection_mask: n_batch, n_pixel
        occ: n_batch, n_pixel
        wpts: n_batch, n_pixel, n_sample, 3
        z_vals: n_pixel, n_sample
        ray_o: n_batch, n_pixel, 3
        ray_d: n_batch * n_pixel, 3
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]

        with torch.no_grad():
            mask = (intersection_mask == True) * (occ == 1)
            full_surf_pts = torch.zeros([n_batch, mask.size(1), 3]).to(wpts)
            full_surf_z = torch.zeros([n_batch, mask.size(1)]).to(wpts)
            full_surf_normal = torch.zeros_like(full_surf_pts)

            if mask.sum() != 0:
                wpts = wpts[mask]
                sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)[mask]
                z_vals = z_vals.view(n_batch, n_pixel, n_sample)[mask]
                ray_o = ray_o.view(n_batch, n_pixel, 3)[mask]
                ray_d = ray_d.view(n_batch, n_pixel, 3)[mask]

                sdf_decoder = lambda wpts_val: self.net.calculate_wpts_sdf(
                    wpts_val, batch)
                surf_pts, surf_z, surf_mask = sphere_tracing(
                    wpts, sdf, z_vals, ray_o, ray_d, sdf_decoder)
                mask[mask] = surf_mask

        if mask.sum() != 0:
            surf_normal = self.net.wpts_gradient(surf_pts, batch)
            full_surf_pts[mask] = surf_pts
            full_surf_z[mask] = surf_z
            full_surf_normal[mask] = surf_normal
        else:
            surf_normal = torch.zeros([n_batch, 0, 3]).to(wpts)

        ret.update({
            'surf_pts': full_surf_pts.view(n_batch, -1, 3),
            'surf_z': full_surf_z.view(n_batch, -1),
            'surf_normal': full_surf_normal.view(n_batch, -1, 3),
            'surf_mask': mask.view(n_batch, -1)
        })

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, occ, batch):
        n_batch = ray_o.shape[0]

        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)
        n_batch, n_pixel, n_sample = wpts.shape[:3]

        # with torch.no_grad():
        #     sdf = self.net.calculate_wpts_sdf(wpts.view(-1, 3), batch)
        #     sdf = sdf.view(n_batch, n_pixel, n_sample)
        #     curr_msk, curr_z_ind = get_intersection_mask(sdf, z_vals)
        #     curr_msk = curr_msk * (occ == 1)

        #     # sample around surface points
        #     wpts, z_vals = self.sample_around_surface(curr_msk, curr_z_ind,
        #                                               z_vals, ray_o, ray_d,
        #                                               near, far, batch)

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        raw_decoder = lambda wpts_val, viewdir_val, dists_val: self.net(
            wpts_val, viewdir_val, dists_val, batch)

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        if cfg.train_mask_mlp:
            if cfg.train_bgfg:
                rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                    raw, z_vals, cfg.white_bkgd)
            else:
                rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs_nerf(
                    raw, z_vals, ray_d, cfg.white_bkgd)
        else:
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        ret.update({
            # 'rgb_map': rgb_map,
            'acc_map': acc_map,
            'depth_map': depth_map,
            'raw': raw.view(n_batch, -1, 4)
        })

        if 'sdf' in ret:
            ret.update({'sdf': ret['sdf'].view(n_batch, -1, 1)})

        if 'resd' in ret:
            resd = ret['resd'].view(n_batch, -1, 3)
            ret.update({'resd': resd})

        if 'fw_resd' in ret:
            fw_resd = ret['fw_resd'].view(n_batch, -1, 3)
            bw_resd = ret['bw_resd'].view(n_batch, -1, 3)
            ret.update({'fw_resd': fw_resd, 'bw_resd': bw_resd})

        if 'pose_pts' in ret:
            pose_pts = ret['pose_pts'].view(n_batch, -1, 3)
            pose_pts_pred = ret['pose_pts_pred'].view(n_batch, -1, 3)
            ret.update({'pose_pts': pose_pts, 'pose_pts_pred': pose_pts_pred})

        if 'pred_pbw' in ret:
            num_joints = ret['pred_pbw'].shape[-1]
            pred_pbw = ret['pred_pbw'].view(n_batch, -1, num_joints)
            smpl_tbw = ret['smpl_tbw'].view(n_batch, -1, num_joints)
            ret.update({'pred_pbw': pred_pbw, 'smpl_tbw': smpl_tbw})

        if 'pbw' in ret:
            num_joints = ret['pbw'].shape[-1]
            pbw = ret['pbw'].view(n_batch, -1, num_joints)
            ret.update({'pbw': pbw})

        if 'tbw' in ret:
            num_joints = ret['tbw'].shape[-1]
            tbw = ret['tbw'].view(n_batch, -1, num_joints)
            ret.update({'tbw': tbw})

        if 'gradients' in ret:
            gradients = ret['gradients'].view(n_batch, -1, 3)
            ret.update({'gradients': gradients})

        if 'full_body_gradients' in ret:
            gradients = ret['full_body_gradients'].view(n_batch, -1, 3)
            ret.update({'full_body_gradients': gradients})

        if 'observed_gradients' in ret:
            ogradients = ret['observed_gradients'].view(n_batch, -1, 3)
            ret.update({'observed_gradients': ogradients})

        if 'resd_jacobian' in ret:
            jac = ret['resd_jacobian'].view(n_batch, -1, 3, 3)
            ret.update({'resd_jacobian': jac})

        if 'raw_fg' in ret:
            # calculate the accumulated transmittance
            alpha = raw[:, :, -1]
            weights = torch.cumprod(torch.cat(
                [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
                -1), -1)[:, :-1]
            acc_alpha = weights[:, -1:]
            ret.update({'acc_alpha': acc_alpha[None]})
            chunk_index = batch['chunk_index']
            if cfg.train_part != 'combine':
                pass
            else:
                import ipdb; ipdb.set_trace(context=11)
                if cfg.train_view_cfg.vis_type == 'full':
                    rgb_map = rgb_map + (acc_alpha[None]) * batch['bg_rgb'][:, chunk_index[0]:chunk_index[1]]
                elif cfg.train_view_cfg.vis_type == 'fg':
                    pass
        ret.update({'rgb_map': rgb_map})


        if 'sdf' in ret:
            # get pixels that outside the mask or no ray-geometry intersection
            sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)
            min_sdf = sdf.min(dim=2)[0]
            free_sdf = min_sdf[occ == 0]
            free_label = torch.zeros_like(free_sdf)

            with torch.no_grad():
                intersection_mask, _ = get_intersection_mask(sdf, z_vals)
            ind = (intersection_mask == False) * (occ == 1)
            sdf = min_sdf[ind]
            label = torch.ones_like(sdf)

            sdf = torch.cat([sdf, free_sdf])
            label = torch.cat([label, free_label])
            ret.update({
                'msk_sdf': sdf.view(n_batch, -1),
                'msk_label': label.view(n_batch, -1)
            })
            # get predicted mask
            pred_mask = torch.sigmoid(min_sdf * -400)
            ret.update({'pred_mask': pred_mask})

        # get intersected points
        if cfg.train_with_normal and 'iter_step' in batch and batch[
                'iter_step'] > 10000:
            ret = self.get_intersection_point(intersection_mask, occ, wpts,
                                              z_vals, ray_o, ray_d, ret, batch)

        if not rgb_map.requires_grad:
            ret = {k: ret[k].detach().cpu() for k in ret.keys()}

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
        if cfg.vis_train_view:
            chunk = 1024
        else:
            chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            occ_chunk = occ[:, i:i + chunk]
            batch['chunk_index'] = [i, i+chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               occ_chunk, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        if cfg.vis_train_view:
            keys = list(keys)
            keys = [key for key in keys if key != 'observed_gradients']

        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
