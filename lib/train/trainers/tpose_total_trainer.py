import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import torch
from lib.networks.renderer import tpose_renderer
from lib.train import make_optimizer
from . import crit
from lib.utils.if_nerf import if_nerf_net_utils


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = tpose_renderer.Renderer(self.net)
        if cfg.fix_body:
            for param in self.net.tpose_human.body_sdf_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.body_color_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.body_deform_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.body_beta_network.parameters():
                param.requires_grad = False

        if cfg.fix_face:
            for param in self.net.tpose_human.face_sdf_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.face_color_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.face_deform_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.face_beta_network.parameters():
                param.requires_grad = False
        if cfg.fix_handl:
            for param in self.net.tpose_human.handl_sdf_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.handl_color_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.handl_deform_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.handl_beta_network.parameters():
                param.requires_grad = False
        if cfg.fix_handr:
            for param in self.net.tpose_human.handr_sdf_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.handr_color_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.handr_deform_network.parameters():
                param.requires_grad = False
            for param in self.net.tpose_human.handr_beta_network.parameters():
                param.requires_grad = False

        self.bw_crit = torch.nn.functional.smooth_l1_loss
        self.img2mse = lambda x, y: torch.mean((x - y)**2)


    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0
        if 'resd' in ret:
            offset_loss = torch.norm(ret['resd'], dim=2).mean()
            scalar_stats.update({'offset_loss': offset_loss})
            loss += 0.01 * offset_loss

        if 'fw_resd' in ret:
            resd = ret['fw_resd'] + ret['bw_resd']
            fwresd_loss = torch.norm(resd, dim=2).mean()
            scalar_stats.update({'fwresd_loss': fwresd_loss})
            loss += fwresd_loss

        if 'pred_pbw' in ret:
            num_joints = ret['pred_pbw'].shape[-1]
            if batch['part_type'] == 0:
                bw_loss = (ret['pred_pbw'] - ret['smpl_tbw']).pow(2).mean() * num_joints / 24
            elif batch['part_type'] == 2:
                bw_loss = (ret['pred_pbw'] - ret['smpl_tbw']).pow(2).mean() * num_joints / 16
            else:
                if num_joints > 24:
                    bw_loss = (ret['pred_pbw'] - ret['smpl_tbw']).pow(2).mean() * num_joints / 24
                else:
                    bw_loss = (ret['pred_pbw'] - ret['smpl_tbw']).pow(2).mean()
            scalar_stats.update({'tbw_loss': bw_loss})
            loss += bw_loss * cfg.bw_loss_weight
        # if batch['latent_index'].item() < cfg.num_trained_mask and 'msk_sdf' in ret and batch['part_type'] == 0:
        if batch['latent_index'].item() < cfg.num_trained_mask and 'msk_sdf' in ret and ret['msk_sdf'].shape[1] > 0 and not cfg.train_bgfg:
            # 不对脸的图像处理，否则会出现凹痕
            #TODO 取脸的mask以上进行这个操作，取个阈值
            mask_loss = crit.sdf_mask_crit(ret, batch)
            if not mask_loss >= 0:
                import ipdb; ipdb.set_trace(context=11)
            scalar_stats.update({'mask_loss': mask_loss})
            loss += mask_loss

        if 'surf_normal' in ret:
            normal_loss = crit.normal_crit(ret, batch)
            scalar_stats.update({'normal_loss': normal_loss})
            loss += 0.01 * normal_loss

        if 'gradients' in ret:
            gradients = ret['gradients']
            grad_loss = (torch.norm(gradients, dim=2) - 1.0)**2
            grad_loss = grad_loss.mean()
            scalar_stats.update({'grad_loss': grad_loss})
            # loss += 0.1 * grad_loss
            loss += cfg.grad_loss_weigth * grad_loss

        if 'full_body_gradients' in ret:
            gradients = ret['full_body_gradients']
            grad_loss = (torch.norm(gradients, dim=2) - 1.0) ** 2
            grad_loss = grad_loss.mean()
            scalar_stats.update({'fbgrad_loss': grad_loss})
            loss += 0.1 * grad_loss

        if 'smooth_sdf' in ret:
            smooth_sdf = ret['smooth_sdf']
            smooth_sdf_loss = torch.norm(smooth_sdf, dim=1).mean()
            scalar_stats.update({'bsdf_loss': smooth_sdf_loss})
            loss += 0.1 * smooth_sdf_loss

        if 'smooth_normal' in ret and batch['iter_step'] > 10000:
            smooth_normal = ret['smooth_normal']
            smooth_normal_loss = torch.norm(smooth_normal, dim=1).mean()
            scalar_stats.update({'snormal_loss': smooth_normal_loss})
            loss += cfg.normal_loss_weight * smooth_normal_loss

        if 'observed_gradients' in ret:
            ogradients = ret['observed_gradients']
            ograd_loss = (torch.norm(ogradients, dim=2) - 1.0)**2
            ograd_loss = ograd_loss.mean()
            scalar_stats.update({'ograd_loss': ograd_loss})
            loss += 0.1 * ograd_loss

        if 'resd_jacobian' in ret:
            elas_loss = crit.elastic_crit(ret, batch)
            scalar_stats.update({'elas_loss': elas_loss})
            loss += 0.1 * elas_loss

        if 'pbw' in ret:
            bw_loss = self.bw_crit(ret['pbw'], ret['tbw'])
            scalar_stats.update({'bw_loss': bw_loss})
            loss += bw_loss

        if 'acc_alpha' in ret:
            acc_alpha = ret['acc_alpha']
            alpha_prior_loss = torch.mean(
                    torch.log(0.1 + acc_alpha) +
                    torch.log(0.1 + 1. - acc_alpha) - -2.20727)
            scalar_stats.update({'alpha_prior_loss': alpha_prior_loss})
            loss += alpha_prior_loss * cfg.alpha_loss_weight

        if 'prior_sdf' in ret:
            prior_sdf = ret['prior_sdf']
            hprior_loss = F.relu(0.0005-prior_sdf).mean()
            # hprior_loss = F.relu(0.001-prior_sdf).mean()
            scalar_stats.update({'hprior_loss': hprior_loss})
            loss += hprior_loss

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
