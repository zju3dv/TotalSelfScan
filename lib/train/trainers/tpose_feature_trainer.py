import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import tpose_feature_renderer_test
from lib.train import make_optimizer
import torchvision.models.vgg as vgg
from collections import namedtuple


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = tpose_feature_renderer_test.Renderer(self.net)

        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.net.tpose_human.feature_network.parameters():
            param.requires_grad = True

        for param in self.net.image_renderer.parameters():
            param.requires_grad = True

        self.img2mse = lambda x, y: torch.mean(torch.abs(x - y))
        self.perceptual_loss = PerceptualLoss()
        self.seg_crit = nn.BCELoss()

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0

        msk = batch['msk'][:, None].float()
        pred_img = ret['pred_img'] * msk
        img = batch['img'] * msk
        img_loss = self.perceptual_loss(pred_img, img)
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        msk = batch['msk'].float()
        pred_msk = ret['pred_msk']
        seg_loss = self.seg_crit(pred_msk, msk)
        scalar_stats.update({'seg_loss': seg_loss})
        loss += seg_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.model = LossNetwork()
        self.model.cuda()
        self.model.eval()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, x, target):
        x_feature = self.model(x[:, 0:3, :, :])
        target_feature = self.model(target[:, 0:3, :, :])

        feature_loss = (
            self.l1_loss(x_feature.relu1, target_feature.relu1) +
            self.l1_loss(x_feature.relu2, target_feature.relu2)) / 2.0

        l1_loss = self.l1_loss(x, target)
        l2_loss = self.mse_loss(x, target)

        loss = feature_loss + l1_loss + l2_loss

        return loss


class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        '''
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
        '''

        self.layer_name_mapping = {'3': "relu1", '8': "relu2"}

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
            if name == '8':
                break
        LossOutput = namedtuple("LossOutput", ["relu1", "relu2"])
        return LossOutput(**output)
