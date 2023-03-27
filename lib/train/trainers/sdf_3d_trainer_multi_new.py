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
        # only train the body sdf network
        for param in self.net.tpose_human.body_sdf_network.parameters():
            param.requires_grad = True
        for param in self.net.tpose_human.body_beta_network.parameters():
            param.requires_grad = True


        self.sdf_crit = torch.nn.L1Loss()

    def forward(self, batch):
        # calculate sdf
        wpts = batch['wpts'][0]
        tpose = batch['tpose'][0]
        wpts = torch.cat([wpts, tpose], dim=0)
        wpts.requires_grad_()

        sdf_network = self.net.tpose_human.sdf_network

        sdf, gradients = sdf_network(wpts, batch, type='both')
        # calculate normal
        gradients = gradients.squeeze()
        # gradients = sdf_network.gradient(wpts, batch).squeeze()

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

        scalar_stats.update({'loss': loss})
        image_stats = {}

        # copy the parameters of body to other parts
        body_sdf_network_state_dict = self.net.tpose_human.body_sdf_network.state_dict()
        if not cfg.debug.all_video_train:
            self.net.tpose_human.face_sdf_network.load_state_dict(body_sdf_network_state_dict)
            self.net.tpose_human.handl_sdf_network.load_state_dict(body_sdf_network_state_dict)
            self.net.tpose_human.handr_sdf_network.load_state_dict(body_sdf_network_state_dict)
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
