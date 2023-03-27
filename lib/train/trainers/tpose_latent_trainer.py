import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import make_renderer
from lib.train import make_optimizer
import torchvision.models.vgg as vgg
from collections import namedtuple


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = make_renderer(cfg, self.net)
        part_names = ['body', 'face', 'handl', 'handr']
        part_name = part_names[cfg.part_type]
        for param in self.net.parameters():
            param.requires_grad = False
        latent_module = eval('self.net.tpose_human.{}_color_network.color_latent'.format(part_name))
        for param in latent_module.parameters():
            param.requires_grad = True

        self.img2mse = lambda x, y: torch.mean((x - y)**2)

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0

        img_loss = self.img2mse(ret['rgb_map'], ret['rgb_gt'])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
