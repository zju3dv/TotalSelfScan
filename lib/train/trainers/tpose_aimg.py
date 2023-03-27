import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import tpose_aimg_renderer
from lib.train import make_optimizer


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = tpose_aimg_renderer.Renderer(self.net)

        self.bw_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0

        bw_loss0 = self.bw_crit(ret['pbw0'], ret['tbw0'])
        bw_loss1 = self.bw_crit(ret['pbw1'], ret['tbw1'])
        scalar_stats.update({'bw_loss0': bw_loss0, 'bw_loss1': bw_loss1})
        loss += bw_loss0 + bw_loss1

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
