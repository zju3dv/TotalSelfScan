import torch.nn as nn
from lib.config import cfg
import torch
from lib.train import make_optimizer


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.img2mse = lambda x, y: torch.mean((x - y)**2)

    def forward(self, batch):
        ret = self.net(batch)

        scalar_stats = {}
        loss = 0

        loss += ret['loss']

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
