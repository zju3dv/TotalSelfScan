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
        for param in self.net.tpose_human.union_sdf_network.parameters():
            param.requires_grad = True

        self.sdf_crit = torch.nn.L1Loss()

    def forward(self, batch):
        # calculate sdf
        tpose = batch['tpose'][0]
        wpts = tpose
        wpts.requires_grad_()

        sdf_network = self.net.tpose_human.sdf_network
        final_sdf_network = self.net.tpose_human.final_sdf_network

        ori_sdf, ori_grad= sdf_network(wpts, batch, type='gradient')
        union_sdf, union_grad = final_sdf_network(wpts, batch, type='gradient')

        # calculate normal
        gradients = union_grad

        ret = {'sdf': ori_sdf, 'gradients': ori_grad}

        scalar_stats = {}
        loss = 0

        grad_loss = (torch.norm(gradients, dim=1) - 1.0)**2
        grad_loss = grad_loss.mean()
        scalar_stats.update({'grad_loss': grad_loss})
        loss += 0.1 * grad_loss

        sdf_loss = self.sdf_crit(union_sdf, ori_sdf)
        scalar_stats.update({'sdf_loss': sdf_loss})
        loss += sdf_loss

        normal_loss = torch.norm((gradients - ori_grad).abs(), dim=1)
        normal_loss = normal_loss.mean()
        scalar_stats.update({'normal_loss': normal_loss})
        loss += normal_loss

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
