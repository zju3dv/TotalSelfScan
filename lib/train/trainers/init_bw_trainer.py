import torch.nn as nn
from lib.config import cfg
import torch
from lib.train import make_optimizer
from . import crit
from lib.utils.blend_utils import *


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.net.backward_bw.parameters():
            param.requires_grad = True

        self.sdf_crit = torch.nn.L1Loss()

    def forward(self, batch):
        tbounds = batch['tbounds']
        bigpose_pts = get_sampling_points(tbounds)

        with torch.no_grad():
            sdf = self.net.tpose_human.sdf_network(bigpose_pts[0], batch)
            sdf = sdf[:, 1][None]
        norm_th = 0.005

        # smpl blend weights at the tpose space
        smpl_tbw = pts_sample_blend_weights(bigpose_pts, batch['tbw'],
                                            batch['tbounds'])
        smpl_tbw, tnorm = smpl_tbw[:, :24], smpl_tbw[:, 24]
        smpl_tbw = smpl_tbw.transpose(1, 2)[sdf < norm_th]
        smpl_tbw = smpl_tbw.transpose(0, 1)[None].contiguous()
        bigpose_pts = bigpose_pts[sdf < norm_th][None]

        # neural blend weights should be the same as smpl blend weights at the tpose space
        init_tpose = pose_points_to_tpose_points(bigpose_pts, smpl_tbw,
                                                 batch['big_A'])
        init_pose = tpose_points_to_pose_points(init_tpose, smpl_tbw,
                                                batch['A'])
        smpl_pbw = pts_sample_blend_weights(init_pose, batch['pbw'],
                                            batch['pbounds'])
        smpl_pbw = smpl_pbw[:, :24]
        pred_pbw = self.net.backward_bw(init_pose, smpl_pbw,
                                        batch['latent_index'])

        # smpl_tbw0, pred_pbw0 = smpl_tbw, pred_pbw

        # pbounds = batch['pbounds']
        # pose_pts = get_sampling_points(pbounds)

        # norm_th = 0.1

        # # neural blend weights should be the same as smpl blend weights
        # smpl_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
        #                                     batch['pbounds'])
        # smpl_pbw, pnorm = smpl_pbw[:, :24], smpl_pbw[:, 24]
        # smpl_pbw = smpl_pbw.transpose(1, 2)[pnorm < norm_th]
        # smpl_pbw = smpl_pbw.transpose(0, 1)[None].contiguous()

        # pose_pts = pose_pts[pnorm < norm_th][None]
        # pred_pbw = self.net.backward_bw(pose_pts, smpl_pbw,
        #                                 batch['latent_index'])

        # # neural blend weights should be the same as smpl blend weights at the tpose space
        # init_tpose = pose_points_to_tpose_points(pose_pts, pred_pbw,
        #                                          batch['A'])
        # init_bigpose = tpose_points_to_pose_points(init_tpose, pred_pbw,
        #                                            batch['big_A'])
        # smpl_tbw = pts_sample_blend_weights(init_bigpose, batch['tbw'],
        #                                     batch['tbounds'])
        # smpl_tbw, tnorm = smpl_tbw[:, :24], smpl_tbw[:, 24]

        # smpl_tbw1, pred_pbw1 = smpl_tbw, pred_pbw

        # from lib.utils import vis_utils
        # import open3d as o3d
        # pc = vis_utils.get_colored_pc(init_bigpose[0].detach().cpu().numpy(), [0, 1, 0])
        # o3d.visualization.draw_geometries([pc])
        # __import__('ipdb').set_trace()

        ret = {'pred_pbw': pred_pbw}

        scalar_stats = {}
        loss = 0

        # bw_loss = (pred_pbw0 - smpl_tbw0).pow(2).mean()
        # scalar_stats.update({'bw_loss': bw_loss})
        # loss += bw_loss

        bw_loss = (pred_pbw - smpl_tbw).pow(2).mean()
        scalar_stats.update({'tbw_loss': bw_loss})
        loss += bw_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats


def get_sampling_points(bounds):
    sh = bounds.shape
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    N_samples = 1024 * 64
    x_vals = torch.rand([sh[0], N_samples])
    y_vals = torch.rand([sh[0], N_samples])
    z_vals = torch.rand([sh[0], N_samples])
    vals = torch.stack([x_vals, y_vals, z_vals], dim=2)
    vals = vals.to(bounds.device)
    pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
    return pts
