import torch.nn as nn
from lib.config import cfg
import torch
from torch.autograd import Variable
from lib.networks.renderer import tpose_renderer
from lib.train import make_optimizer
from . import crit


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = tpose_renderer.Renderer(self.net)

        self.bw_crit = torch.nn.functional.smooth_l1_loss
        self.img2mse = lambda x, y: torch.mean((x - y)**2)

    def inner_loop(self, batch_list):
        inner_net = self.net.clone()
        inner_renderer = tpose_renderer.Renderer(inner_net)
        inner_optimizer = torch.optim.Adam(inner_net.parameters(), lr=5e-4)
        num_frame = len(batch_list)
        for i in range(32):
            batch = batch_list[i % num_frame]
            ret = inner_renderer.render(batch)

            scalar_stats = {}
            loss = 0

            if 'resd' in ret:
                offset_loss = torch.norm(ret['resd'], dim=2).mean()
                scalar_stats.update({'offset_loss': offset_loss})
                loss += 0.01 * offset_loss

            mask_loss = crit.sdf_mask_crit(ret, batch)
            scalar_stats.update({'mask_loss': mask_loss})
            loss += mask_loss

            if 'gradients' in ret:
                gradients = ret['gradients']
                grad_loss = (torch.norm(gradients, dim=2) - 1.0)**2
                grad_loss = grad_loss.mean()
                scalar_stats.update({'grad_loss': grad_loss})
                loss += 0.1 * grad_loss

            if 'observed_gradients' in ret:
                ogradients = ret['observed_gradients']
                ograd_loss = (torch.norm(ogradients, dim=2) - 1.0) ** 2
                ograd_loss = ograd_loss.mean()
                scalar_stats.update({'ograd_loss': ograd_loss})
                loss += 0.1 * ograd_loss

            mask = batch['mask_at_box']
            img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
            scalar_stats.update({'img_loss': img_loss})
            loss += img_loss

            scalar_stats.update({'loss': loss})

            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        scalar_stats = {
            k: scalar_stats[k].detach()
            for k in scalar_stats.keys()
        }

        return inner_net, scalar_stats

    def forward(self, batch):
        batch_size = len(batch)

        # inner loop
        inner_nets = []
        scalar_stats_list = []
        for i in range(batch_size):
            inner_net, scalar_stats = self.inner_loop(batch[i])
            inner_nets.append(list(inner_net.parameters()))
            scalar_stats_list.append(scalar_stats)

        # inject updates into gradients
        params = list(self.net.parameters())
        for i in range(len(params)):
            p = params[i]
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size()).to(p))

            grad_list = []
            for i_net in range(batch_size):
                grad_list.append(p.data - inner_nets[i_net][i].data)
            grad = torch.stack(grad_list).mean(dim=0)

            p.grad.data.add_(grad)

        keys = scalar_stats_list[0].keys()
        scalar_stats = {
            k: torch.stack([r[k] for r in scalar_stats_list]).mean()
            for k in keys
        }
        loss = scalar_stats['loss']

        ret = {}
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
