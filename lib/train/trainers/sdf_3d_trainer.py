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

        if 'tpose_human' in dir(self.net):
            for param in self.net.tpose_human.sdf_network.parameters():
                param.requires_grad = True
        else:
            for param in self.net.sdf_network.parameters():
                param.requires_grad = True

        self.sdf_crit = torch.nn.L1Loss()

    def forward(self, batch):
        # calculate sdf
        wpts = batch['wpts'][0]
        tpose = batch['tpose'][0]
        wpts = torch.cat([wpts, tpose], dim=0)
        wpts.requires_grad_()

        if 'tpose_human' in dir(self.net):
            sdf_network = self.net.tpose_human.sdf_network
        else:
            sdf_network = self.net.sdf_network

        sdf_nn_output = sdf_network(wpts, batch)

        sdf = sdf_nn_output[:, 0]
        feature_vector = sdf_nn_output[:, 1:]

        # calculate normal
        gradients = sdf_network.gradient(wpts, batch).squeeze()

        ret = {'sdf': sdf, 'gradients': gradients}

        scalar_stats = {}
        loss = 0

        grad_loss = (torch.norm(gradients, dim=1) - 1.0)**2
        grad_loss = grad_loss.mean()
        scalar_stats.update({'grad_loss': grad_loss})
        loss += 0.1 * grad_loss

        sdf = sdf[:len(batch['sdf'][0])]
        gradients = gradients[:len(batch['sdf'][0])]

        sdf_loss = self.sdf_crit(sdf, batch['sdf'][0])
        scalar_stats.update({'sdf_loss': sdf_loss})
        loss += sdf_loss

        normal_loss = torch.norm((gradients - batch['normal'][0]).abs(), dim=1)
        normal_loss = normal_loss.mean()
        scalar_stats.update({'normal_loss': normal_loss})
        loss += normal_loss

        if batch['meta']['part'] == 'body':
            tbounds = batch['tbounds']
            bigpose_pts = get_sampling_points(tbounds)
            with torch.no_grad():
                sdf = self.net.tpose_human.sdf_network(bigpose_pts[0], batch)
                sdf = sdf[:, 1][None]
            norm_th = 0.005
            sdf_ind = (sdf < norm_th) * (sdf.abs() < 0.1)

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
