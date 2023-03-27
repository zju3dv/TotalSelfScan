import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import sdf_renderer
from lib.train import make_optimizer
from . import crit


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = sdf_renderer.Renderer(self.net)

        self.bw_crit = torch.nn.functional.smooth_l1_loss
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)

        for param in self.net.parameters():
            param.requires_grad = False

        if 'tpose_human' in dir(self.net):
            for param in self.net.tpose_human.parameters():
                param.requires_grad = True
        else:
            for param in self.net.parameters():
                param.requires_grad = True

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0

        gradients = ret['gradients']
        grad_loss = (torch.norm(gradients, dim=2) - 1.0) ** 2
        grad_loss = grad_loss.mean()
        scalar_stats.update({'grad_loss': grad_loss})
        loss += 0.1 * grad_loss

        # mask_loss = crit.sdf_mask_crit(ret, batch)
        # scalar_stats.update({'mask_loss': mask_loss})
        # loss += mask_loss

        mask = batch['mask_at_box']
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
