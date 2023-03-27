import torch.nn as nn
from lib.config import cfg
import torch
from lib.train import make_optimizer
from . import crit
from lib.utils.if_nerf import if_nerf_net_utils


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.net.forward_resd.parameters():
            param.requires_grad = True

    def forward(self, batch):
        ret = {}
        tpose = batch['tpose']
        latent_index = batch['latent_index']

        fw_resd = self.net.forward_resd(tpose, latent_index)
        fw_tpose = tpose + fw_resd
        bw_resd = self.net.calculate_residual_deformation(
            fw_tpose, latent_index)

        scalar_stats = {}
        loss = 0

        resd = fw_resd + bw_resd
        fwresd_loss = torch.norm(resd, dim=2).mean()
        scalar_stats.update({'fwresd_loss': fwresd_loss})
        loss += fwresd_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
