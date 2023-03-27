import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import tpose_forward_renderer
from lib.train import make_optimizer
import torchvision.models.vgg as vgg
from collections import namedtuple


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = tpose_forward_renderer.Renderer(self.net)

        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.net.tpose_human.color_network.parameters():
            param.requires_grad = True

        self.img2mse = lambda x, y: torch.mean((x - y)**2)

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0

        mask = ret['render_mask'][batch['mask_at_box']][None]
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
