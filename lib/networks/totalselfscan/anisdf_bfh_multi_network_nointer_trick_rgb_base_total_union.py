import torch.nn as nn
#import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from lib.datasets import embedder
from lib.utils import net_utils
from lib.utils.total_utils_base import judge_bounds, judge_bounds_inter, PerPartCompute, get_near_far
from lib.utils.debugger import dbg


class Network(nn.Module):
    def __init__(self, init_network=True):
        super(Network, self).__init__()

        self.tpose_human = TPoseHuman()

        self.resd_latent = nn.Embedding(cfg.num_latent_code, 128)

        self.actvn = nn.ReLU()
        self.part = ['body', 'face', 'handl', 'handr']

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.resd_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.resd_fc = nn.Conv1d(W, 3, 1)
        self.resd_fc.bias.data.fill_(0)

        self.backward_bw = BackwardBlendWeight(num_joints=52)
        # self.backward_bw_hand = BackwardBlendWeight(num_joints=16)

        if cfg.has_forward_resd:
            self.forward_resd = ForwardResd()
        if init_network and 'init_sdf' in cfg:
            net_utils.load_network(self,
                                   'data/trained_model/{}/'.format(cfg.task) + cfg.init_sdf,
                                   strict=False)

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def calculate_residual_deformation(self, tpose, latent_index, batch):
        latents = self.resd_latent
        features = self.get_point_feature(tpose, latent_index, latents)[0].transpose(1, 0)

        funcs = {0: self.tpose_human.body_deform_network,
                 1: self.tpose_human.face_deform_network,
                 2: self.tpose_human.handl_deform_network,
                 3: self.tpose_human.handr_deform_network}
        func_args = {
            'index': {
                'features': features
            },
            'noindex': {}
        }
        resd = PerPartCompute(tpose[0], batch, funcs, func_args)[None]

        return resd

    def pose_points_to_tpose_points(self, pose_pts, pose_dirs, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        with torch.no_grad():
            init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                                batch['pbounds'])
            init_pbw, pnorm = init_pbw[:, :-1], init_pbw[:, -1]
            pred_pbw = self.backward_bw(pose_pts, init_pbw,
                                        batch['latent_index'])

        # transform points from i to i_0
        init_tpose = pose_points_to_tpose_points(pose_pts, pred_pbw,
                                                 batch['A'])
        init_bigpose = tpose_points_to_pose_points(init_tpose, pred_pbw,
                                                   batch['big_A'])
        resd = self.calculate_residual_deformation(init_bigpose,
                                                   batch['latent_index'], batch)
        tpose = init_bigpose + resd

        if cfg.tpose_viewdir and pose_dirs is not None:
            init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, pred_pbw,
                                                 batch['A'])
            tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, pred_pbw,
                                                 batch['big_A'])
        else:
            tpose_dirs = None

        return tpose, tpose_dirs, init_bigpose, resd

    def calculate_bigpose_smpl_bw(self, bigpose, input_bw):
        smpl_bw = pts_sample_blend_weights(bigpose, input_bw['tbw'],
                                           input_bw['tbounds'])
        return smpl_bw

    def calculate_wpts_sdf(self, wpts, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        # transform points from the pose space to the tpose space
        tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(
            pose_pts, None, batch)
        tpose = tpose[0]
        sdf = self.tpose_human.sdf_network(tpose, batch)[:, :1]

        return sdf

    def wpts_gradient(self, wpts, batch):
        wpts.requires_grad_(True)
        with torch.enable_grad():
            sdf = self.calculate_wpts_sdf(wpts, batch)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients

    def gradient_of_deformed_sdf(self, x, batch):
        x.requires_grad_(True)
        with torch.enable_grad():
            resd = self.calculate_residual_deformation(x,
                                                       batch['latent_index'],
                                                       batch)
            tpose = x + resd
            tpose = tpose[0]
            y = self.tpose_human.sdf_network(tpose, batch)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients, y[None]

    def forward(self, wpts, viewdir, dists, batch):
        if batch['part_type'] == 0:
            # body part
            # transform points from the world space to the pose space
            wpts = wpts[None]
            pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
            viewdir = viewdir[None]
            pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])

            with torch.no_grad():
                init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                                    batch['pbounds'])
                pnorm = init_pbw[:, -1]
                norm_th = 0.1
                pind = pnorm < norm_th
                pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
                pose_pts = pose_pts[pind][None]
                viewdir = viewdir[pind][None]
                pose_dirs = pose_dirs[pind][None]

            # transform points from the pose space to the tpose space
            tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(
                pose_pts, pose_dirs, batch)
            tpose = tpose[0]
            if cfg.tpose_viewdir:
                viewdir = tpose_dirs[0]
            else:
                viewdir = viewdir[0]
            ret = self.tpose_human(tpose, viewdir, dists, batch)
            ind = ret['sdf'][:, 0].detach().abs() < 0.02
            init_bigpose = init_bigpose[0][ind][None].detach().clone()

            if ret['raw'].requires_grad and ind.sum() != 0:
                observed_gradients, _ = self.gradient_of_deformed_sdf(
                    init_bigpose, batch)
                ret.update({'observed_gradients': observed_gradients})

            tbounds = batch['tbounds'][0]
            inside = tpose > tbounds[:1]
            inside = inside * (tpose < tbounds[1:])
            outside = torch.sum(inside, dim=1) != 3
            ret['raw'][outside] = 0

            n_batch, n_point = wpts.shape[:2]
            raw = torch.zeros([n_batch, n_point, 4]).to(wpts)
            raw[pind] = ret['raw']
            sdf = 10 * torch.ones([n_batch, n_point, 1]).to(wpts)
            sdf[pind] = ret['sdf']
            ret.update({'raw': raw, 'sdf': sdf})

            ret.update({'resd': resd})

            # neural blend weights should be the same as smpl blend weights at the tpose space
            if ret['raw'].requires_grad and cfg.bw_loss_weight > 0:
                pred_pbw, smpl_tbw = sample_blend_weights(
                    self, batch)
                pred_pbw = pred_pbw.transpose(1, 2)
                smpl_tbw = smpl_tbw.transpose(1, 2)
                ret.update({
                    'pred_pbw': pred_pbw,
                    'smpl_tbw': smpl_tbw
                })
        elif batch['part_type'] == 1:
            # face part
            # transform points from the world space to the pose space
            wpts = wpts[None]
            n_batch, n_point = wpts.shape[:2]

            viewdir = viewdir[None]
            with torch.no_grad():
                pnorm = pts_sample_blend_weights(wpts, batch['face_dist'],
                                                 batch['tbounds'])
                pnorm = pnorm[:, 0]
                norm_th = 0.04
                pind = pnorm < norm_th
                pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
                wpts = wpts[pind][None]
                viewdir = viewdir[pind][None]
            resd = self.calculate_residual_deformation(wpts,
                                                       batch['latent_index'], batch)
            tpose = wpts + resd
            tpose = tpose[0]
            viewdir = viewdir[0]
            ret = self.tpose_human(tpose, viewdir, dists, batch)
            ind = ret['sdf'][:, 0].detach().abs() < 0.02
            init_bigpose = wpts[0][ind][None].detach().clone()

            if ind.sum() != 0:
                observed_gradients, _ = self.gradient_of_deformed_sdf(
                    init_bigpose, batch)
                ret.update({'observed_gradients': observed_gradients})

            raw = torch.zeros([n_batch, n_point, 4]).to(wpts)
            raw[pind] = ret['raw']
            sdf = 10 * torch.ones([n_batch, n_point, 1]).to(wpts)
            sdf[pind] = ret['sdf']
            ret.update({'raw': raw, 'sdf': sdf})

        elif batch['part_type'] == 2:
            # transform points from the world space to the pose space
            wpts = wpts[None]
            pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
            viewdir = viewdir[None]
            pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])

            with torch.no_grad():
                init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                                    batch['pbounds'])
                pnorm = init_pbw[:, -1]
                norm_th = 0.1 #0.015
                pind = pnorm < norm_th
                pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
                pose_pts = pose_pts[pind][None]
                viewdir = viewdir[pind][None]
                pose_dirs = pose_dirs[pind][None]

            # transform points from the pose space to the tpose space
            tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(
                pose_pts, pose_dirs, batch)
            tpose = tpose[0]
            if cfg.tpose_viewdir:
                viewdir = tpose_dirs[0]
            else:
                viewdir = viewdir[0]
            ret = self.tpose_human(tpose, viewdir, dists, batch)
            ind = ret['sdf'][:, 0].detach().abs() < 0.01
            init_bigpose = init_bigpose[0][ind][None].detach().clone()

            if ret['raw'].requires_grad and ind.sum() != 0:
                observed_gradients, _ = self.gradient_of_deformed_sdf(
                    init_bigpose, batch)
                ret.update({'observed_gradients': observed_gradients})

            tbounds = batch['tbounds'][0]
            inside = tpose > tbounds[:1]
            inside = inside * (tpose < tbounds[1:])
            outside = torch.sum(inside, dim=1) != 3
            ret['raw'][outside] = 0

            n_batch, n_point = wpts.shape[:2]
            raw = torch.zeros([n_batch, n_point, 4]).to(wpts)
            raw[pind] = ret['raw']
            sdf = 10 * torch.ones([n_batch, n_point, 1]).to(wpts)
            sdf[pind] = ret['sdf']
            ret.update({'raw': raw, 'sdf': sdf})

            ret.update({'resd': resd})

            # neural blend weights should be the same as smpl blend weights at the tpose space
            if ret['raw'].requires_grad and cfg.bw_loss_weight > 0:
                pred_pbw, smpl_tbw = sample_blend_weights(
                    self, batch)
                pred_pbw = pred_pbw.transpose(1, 2)
                smpl_tbw = smpl_tbw.transpose(1, 2)
                ret.update({
                    'pred_pbw': pred_pbw,
                    'smpl_tbw': smpl_tbw
                })

        return ret


def sample_blend_weights(net, batch):
    if batch['part_type'] == 0 or batch['part_type'] == 1:

        tbounds = batch['tbounds']
        bigpose_pts = get_sampling_points(tbounds, N_samples=1024 * 64)

        with torch.no_grad():
            sdf = net.tpose_human.sdf_network(bigpose_pts[0], batch)[:, 0]

        sdf = sdf[None]
        sdf_ind = (sdf < 0.005) * (sdf.abs() < 0.02)
        sdf_ind[torch.arange(len(sdf)), torch.argmin(sdf.abs(), dim=1)] = True

        bigpose_pts = bigpose_pts[sdf_ind][None]

        # smpl blend weights at the tpose space
        smpl_tbw = pts_sample_blend_weights(bigpose_pts, batch['tbw'],
                                            batch['tbounds'])
        smpl_tbw, tnorm = smpl_tbw[:, :-1], smpl_tbw[:, -1]

        # neural blend weights should be the same as smpl blend weights at the tpose space
        init_tpose = pose_points_to_tpose_points(bigpose_pts, smpl_tbw,
                                                 batch['big_A'])
        init_pose = tpose_points_to_pose_points(init_tpose, smpl_tbw, batch['A'])
        smpl_pbw = pts_sample_blend_weights(init_pose, batch['pbw'],
                                            batch['pbounds'])
        smpl_pbw = smpl_pbw[:, :-1]
        pred_pbw = net.backward_bw(init_pose, smpl_pbw, batch['latent_index'])

    elif batch['part_type'] == 2:
        tbounds = batch['tbounds']
        bigpose_pts = get_sampling_points(tbounds, N_samples=1024 * 64)

        with torch.no_grad():
            sdf = net.tpose_human.sdf_network(bigpose_pts[0], batch)[:, 0]

        sdf = sdf[None]
        sdf_ind = (sdf < 0.003) * (sdf.abs() < 0.01)
        sdf_ind[torch.arange(len(sdf)), torch.argmin(sdf.abs(), dim=1)] = True

        bigpose_pts = bigpose_pts[sdf_ind][None]

        # smpl blend weights at the tpose space
        smpl_tbw = pts_sample_blend_weights(bigpose_pts, batch['tbw'],
                                            batch['tbounds'])
        smpl_tbw, tnorm = smpl_tbw[:, :-1], smpl_tbw[:, -1]

        # neural blend weights should be the same as smpl blend weights at the tpose space
        init_tpose = pose_points_to_tpose_points(bigpose_pts, smpl_tbw,
                                                 batch['big_A'])
        init_pose = tpose_points_to_pose_points(init_tpose, smpl_tbw, batch['A'])
        smpl_pbw = pts_sample_blend_weights(init_pose, batch['pbw'],
                                            batch['pbounds'])
        smpl_pbw = smpl_pbw[:, :-1]
        pred_pbw = net.backward_bw(init_pose, smpl_pbw, batch['latent_index'])
    return pred_pbw, smpl_tbw


class BackwardBlendWeight(nn.Module):
    def __init__(self, num_joints=24):
        super(BackwardBlendWeight, self).__init__()

        self.bw_latent = nn.Embedding(cfg.num_latent_code, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, num_joints, 1)

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def forward(self, ppts, smpl_bw, latent_index):
        latents = self.bw_latent
        features = self.get_point_feature(ppts, latent_index, latents)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw


class ForwardResd(nn.Module):
    def __init__(self):
        super(ForwardResd, self).__init__()

        self.resd_latent = nn.Embedding(cfg.num_latent_code, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.resd_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.resd_fc = nn.Conv1d(W, 3, 1)
        self.resd_fc.bias.data.fill_(0)

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def forward(self, tpose, latent_index):
        latents = self.resd_latent
        features = self.get_point_feature(tpose, latent_index, latents)
        net = features
        for i, l in enumerate(self.resd_linears):
            net = self.actvn(self.resd_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        resd = self.resd_fc(net)
        resd = resd.transpose(1, 2)
        resd = 0.05 * torch.tanh(resd)
        return resd


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        self.body_sdf_network = SDFNetwork()
        self.face_sdf_network = SDFNetwork(layers=cfg.fh_layers)
        self.handl_sdf_network = SDFNetwork(layers=cfg.fh_layers)
        self.handr_sdf_network = SDFNetwork(layers=cfg.fh_layers)
        self.body_beta_network = BetaNetwork()
        self.face_beta_network = BetaNetwork()
        self.handl_beta_network = BetaNetwork()
        self.handr_beta_network = BetaNetwork()
        self.body_color_network = ColorNetwork()
        self.face_color_network = ColorNetwork()
        self.handl_color_network = ColorNetwork()
        self.handr_color_network = ColorNetwork()
        self.color_network = ColorNetwork()
        self.body_deform_network = DeformNetwork('body')
        self.face_deform_network = DeformNetwork('face')
        self.handl_deform_network = DeformNetwork('handl')
        self.handr_deform_network = DeformNetwork('handr')
        self.union_sdf_network = CombineNetwork()

    def sdf_to_alpha(self, sdf, beta):
        x = -sdf

        # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
        ind0 = x <= 0
        val0 = 1 / beta * (0.5 * torch.exp(x[ind0] / beta))

        # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
        ind1 = x > 0
        val1 = 1 / beta * (1 - 0.5 * torch.exp(-x[ind1] / beta))

        val = torch.zeros_like(sdf)
        val[ind0] = val0
        val[ind1] = val1

        return val

    def get_inside(self, pts, bound):
        inside = pts > bound[:1]
        inside = inside * (pts < bound[1:])
        inside = torch.sum(inside, dim=1) == 3

        return inside

    def forward(self, wpts, viewdir, dists, batch, ori_viewdir=None):
        # classify points to each bbox
        if batch['part_type'] == 2:
            pass
            # import ipdb; ipdb.set_trace(context=11)
        face_only_ind = self.get_inside(wpts, batch['tbounds_face'][0])
        handl_only_ind = self.get_inside(wpts, batch['tbounds_handl'][0])
        handr_only_ind = self.get_inside(wpts, batch['tbounds_handr'][0])
        body_only_ind = ~face_only_ind * ~handl_only_ind * ~handr_only_ind
        # calculate sdf
        wpts_canonical = wpts.clone()
        if cfg.debug.transform_face:
            wpts[face_only_ind] = wpts[face_only_ind] @ batch['R_canonical2face'][0].t() + \
                                   batch['T_canonical2face'][0]
        wpts.requires_grad_()
        with torch.enable_grad():
            funcs = {0: self.body_sdf_network,
                     # 1: self.body_sdf_network,
                     1: self.face_sdf_network,
                     2: self.handl_sdf_network,
                     3: self.handr_sdf_network}
            func_args = {
                'index':{
                    'inputs': wpts
                },
                'noindex':{
                    'batch': batch
                }
            }
            sdf_nn_output = PerPartCompute(wpts_canonical, batch, funcs, func_args)
            sdf = sdf_nn_output[:, :1]
            # calculate smooth sdf
            full_body_sdf_nn_output = self.body_sdf_network(wpts_canonical, batch)
            full_body_sdf = full_body_sdf_nn_output[:, :1]
            bound_id = judge_bounds(wpts_canonical, batch)
            body_only_ind = (bound_id == 0)
            smooth_sdf = full_body_sdf[~body_only_ind] - sdf[~body_only_ind]
            # add overlap sdf loss
            if cfg.train_with_overlap:
                pts_bf_overlap = batch['pts_bf_overlap'][0].float()
                pts_bhl_overlap = batch['pts_bhl_overlap'][0].float()
                overlap_body_sdf_nn_output = self.body_sdf_network(torch.cat([pts_bf_overlap, pts_bhl_overlap], dim=0), batch)
                overlap_face_sdf_nn_output = self.face_sdf_network(pts_bf_overlap, batch)
                overlap_handl_sdf_nn_output = self.handl_sdf_network(pts_bhl_overlap, batch)
                overlap_handr_sdf_nn_output = self.handr_sdf_network(pts_bhl_overlap, batch)
                overlap_smooth_sdf = overlap_body_sdf_nn_output[:, :1] - torch.cat([overlap_face_sdf_nn_output,
                                                                                    overlap_handl_sdf_nn_output,
                                                                                    overlap_handr_sdf_nn_output],
                                                                                    dim=0)[:, :1]
                smooth_sdf = torch.cat([smooth_sdf, overlap_smooth_sdf])

        feature_vector = sdf_nn_output[:, 1:]

        # calculate normal
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        if cfg.trick_sample and not cfg.vis_train_view and not cfg.final_hand and not cfg.vis_novel_view and not cfg.forward_rendering:
            full_body_gradients = torch.autograd.grad(outputs=full_body_sdf[~body_only_ind],
                                            inputs=wpts,
                                            grad_outputs=d_output[~body_only_ind],
                                            create_graph=True,
                                            retain_graph=True,
                                            # allow_unused=True,
                                            only_inputs=True)[0]
        else:
            full_body_gradients = gradients
        # gradients = self.sdf_network.gradient(wpts, batch)[:, 0]
        # calculate alpha
        wpts = wpts.detach()
        betas = [self.body_beta_network(wpts).clamp(1e-9, 1e6),
                 # self.body_beta_network(wpts).clamp(1e-9, 1e6),
                 self.face_beta_network(wpts).clamp(1e-9, 1e6),
                 self.handl_beta_network(wpts).clamp(1e-9, 1e6),
                 self.handr_beta_network(wpts).clamp(1e-9, 1e6)]
        funcs = {0: self.sdf_to_alpha,
                 1: self.sdf_to_alpha,
                 2: self.sdf_to_alpha,
                 3: self.sdf_to_alpha}
        func_args = {
            'index': {
                'sdf': sdf
            },
            'noindex': {},
            'step': {
                'beta': betas
            }
        }
        alpha = PerPartCompute(wpts_canonical, batch, funcs, func_args)
        #
        # beta = self.body_beta_network(wpts).clamp(1e-9, 1e6)
        # alpha = self.sdf_to_alpha(sdf[body_only_ind], beta)

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * 0.005)
        alpha = raw2alpha(alpha[:, 0], dists)

        # calculate color
        ind = batch['latent_index']
        total_group_fix = batch['total_group_fix']
        if True:
            # after latent optim, then use this
            total_group_fix[0, 0] = 0
            total_group_fix[0, 1] = 0
            total_group_fix[0, 2] = 0
            total_group_fix[0, 3] = 0
        if cfg.debug.transform_face:
            viewdir_transform_face = viewdir.clone()
            if not cfg.select_view:
                viewdir_transform_face[face_only_ind] = viewdir[face_only_ind] @ batch['R_canonical2face'][0].t()
        ind = torch.zeros_like(ind)
        funcs = {0: self.body_color_network,
                 1: self.face_color_network,
                 2: self.handl_color_network,
                 3: self.handr_color_network}
        func_args = {
            'index': {
                'points': wpts,
                'normals': gradients,
                'view_dirs': viewdir_transform_face if cfg.debug.transform_face else viewdir,
                'feature_vectors': feature_vector
            },
            'noindex': {},
            #     'latent_index': ind
            # },
            'step': {
                'latent_index': total_group_fix.t()
                }
            }
        rgb = PerPartCompute(wpts_canonical, batch, funcs, func_args)
        if cfg.process.smooth_rgb:
            if ori_viewdir == None:
                ori_viewdir = viewdir
            body_gradients = self.body_sdf_network.gradient(wpts_canonical, batch).squeeze()
            full_body_rgb = self.body_color_network(wpts_canonical, body_gradients, ori_viewdir,
                                                    full_body_sdf_nn_output[:, 1:], ind)
            bound_id = judge_bounds_inter(wpts_canonical, batch)
            # left hand smooth
            body_handl_ind = (bound_id == 5)
            if body_handl_ind.sum() > 0:
                length = cfg.data.body_hand_bound_overlap
                weight = wpts_canonical[body_handl_ind][:, :1] - batch['tbounds_handl'][0, 0, 0]
                weight /= length
                rgb[body_handl_ind] = rgb[body_handl_ind] * weight + full_body_rgb[
                    body_handl_ind] * (1 - weight)
            # right hand smooth
            body_handr_ind = (bound_id == 6)
            if body_handr_ind.sum() > 0:
                length = cfg.data.body_hand_bound_overlap
                weight = wpts_canonical[body_handr_ind][:, :1] - batch['tbounds_body'][0, 0, 0]
                weight /= length
                rgb[body_handr_ind] = rgb[body_handr_ind] * (1-weight) + full_body_rgb[
                    body_handr_ind] * (weight)

            # body face smooth
            # import ipdb; ipdb.set_trace(context=11)
            # from lib.utils.debugger import dbg
            # dbg.showL3D([wpts_canonical[body_face_ind].cpu(), wpts[body_face_ind].cpu()])
            body_face_ind = (bound_id == 4)
            if body_face_ind.sum() > 0:
                length = cfg.data.body_face_bound_overlap
                weight = wpts_canonical[body_face_ind][:, 1:2] - batch['tbounds_face'][0, 0, 1]
                weight /= length
                # print('minimal weight is {}'.format(weight.min().item()))
                rgb[body_face_ind] = rgb[body_face_ind] * weight + full_body_rgb[
                    body_face_ind] * (1 - weight)


        raw = torch.cat((rgb, alpha[:, None]), dim=1)
        if cfg.trick_sample and not cfg.vis_train_view and not cfg.final_hand and not cfg.vis_novel_view and not cfg.forward_rendering:
            ret = {'raw': raw, 'sdf': sdf, 'smooth_sdf': smooth_sdf, 'gradients': gradients,
                   'full_body_gradients': full_body_gradients[~body_only_ind]}
        else:
            ret = {'raw': raw, 'sdf': sdf, 'gradients': gradients}

        return ret

    def sdf_network(self, wpts, batch, type='sdf', multi=True):
        # classify points to each bbox
        face_only_ind = self.get_inside(wpts, batch['tbounds_face'][0])
        wpts_canonical = wpts.clone()
        if cfg.debug.transform_face:
            wpts[face_only_ind] = wpts[face_only_ind] @ batch['R_canonical2face'][0].t() + \
                                   batch['T_canonical2face'][0]
        wpts.requires_grad_()
        with torch.enable_grad():
            funcs = {0: self.body_sdf_network,
                     # 1: self.body_sdf_network,
                     1: self.face_sdf_network,
                     2: self.handl_sdf_network,
                     3: self.handr_sdf_network}
            func_args = {
                'index':{
                    'inputs': wpts
                },
                'noindex':{
                    'batch': batch
                }
            }
            sdf_nn_output = PerPartCompute(wpts_canonical, batch, funcs, func_args)
        if not multi:
            full_body_sdf_nn_output = self.body_sdf_network(wpts, batch)
            sdf_nn_output = full_body_sdf_nn_output

        if type == 'sdf':
            return sdf_nn_output
        else:
            sdf = sdf_nn_output[:, :1]
            # calculate normal
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(outputs=sdf,
                                            inputs=wpts,
                                            grad_outputs=d_output,
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)[0]
            return sdf, gradients

    def final_sdf_network(self, wpts, batch, type='sdf', multi=True):
        # classify points to each bbox
        wpts.requires_grad_()
        with torch.enable_grad():
            body_feature = self.body_sdf_network.last_feature(wpts, batch)
            face_feature = self.face_sdf_network.last_feature(wpts, batch)
            handl_feature = self.handl_sdf_network.last_feature(wpts, batch)
            handr_feature = self.handr_sdf_network.last_feature(wpts, batch)
            full_feature = torch.cat([body_feature, face_feature, handl_feature, handr_feature], dim=1)
            final_sdf_nn_output = self.union_sdf_network(full_feature, batch)

        if type == 'sdf':
            return final_sdf_nn_output
        else:
            sdf = final_sdf_nn_output[:, :1]
            # calculate normal
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(outputs=sdf,
                                            inputs=wpts,
                                            grad_outputs=d_output,
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)[0]
            return sdf, gradients

class CombineNetwork(nn.Module):
    def __init__(self):
        super(CombineNetwork, self).__init__()

        geometric_init = True
        weight_norm = True
        bias = 0.5

        activation = 'softplus'
        self.lin0 = nn.Linear(1024, 256)
        self.lin1 = nn.Linear(256, 257)
        if geometric_init:
            torch.nn.init.normal_(self.lin1.weight,
                                  mean=np.sqrt(np.pi) /
                                       np.sqrt(256),
                                  std=0.0001)
            torch.nn.init.constant_(self.lin1.bias, -bias)

            torch.nn.init.constant_(self.lin0.bias, 0.0)
            torch.nn.init.normal_(self.lin0.weight, 0.0,
                                  np.sqrt(2) / np.sqrt(256))

        if weight_norm:
            self.lin0 = nn.utils.weight_norm(self.lin0)
            self.lin1 = nn.utils.weight_norm(self.lin1)


        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs, batch):
        x = inputs
        x = self.lin0(x)
        x = self.activation(x)
        x = self.lin1(x)
        return x
    #
    # def last_feature(self, inputs, batch):
    #     inputs = inputs * self.scale
    #     if self.embed_fn_fine is not None:
    #         inputs = self.embed_fn_fine(inputs)
    #     x = inputs
    #     for l in range(0, self.num_layers - 2):
    #         lin = getattr(self, "lin" + str(l))
    #
    #         if l in self.skip_in:
    #             x = torch.cat([x, inputs], 1) / np.sqrt(2)
    #
    #         x = lin(x)
    #
    #         if l < self.num_layers - 2:
    #             x = self.activation(x)
    #     return x
    #
    # def sdf(self, x, batch):
    #     return self.forward(x, batch)[:, :1]
    #
    # def gradient(self, x, batch):
    #     x.requires_grad_(True)
    #     with torch.enable_grad():
    #         y = self.sdf(x, batch)
    #     d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    #     gradients = torch.autograd.grad(outputs=y,
    #                                     inputs=x,
    #                                     grad_outputs=d_output,
    #                                     create_graph=True,
    #                                     retain_graph=True,
    #                                     only_inputs=True)[0]
    #     return gradients.unsqueeze(1)

class DeformNetwork(nn.Module):
    def __init__(self, part):
        super(DeformNetwork, self).__init__()
        input_ch = 191
        D = 8
        W = 256
        self.actvn = nn.ReLU()
        self.skips = [4]
        self.resd_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.resd_fc = nn.Conv1d(W, 3, 1)
        self.resd_fc.bias.data.fill_(0)

        if part == 'body' or part == 'face':
            self.c = 0.05
        elif part == 'handl' or part == 'handr':
            self.c = 0.005

    def unit_test(self, features, resd_linears, resd_fc):
        net = features
        for i, l in enumerate(resd_linears):
            net = self.actvn(resd_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        resd = resd_fc(net)
        resd = resd.transpose(1, 2)
        resd = self.c * torch.tanh(resd)

        return resd

    def forward(self, features):
        features = features.transpose(1, 0)
        features = features[None, ...]
        net = features
        for i, l in enumerate(self.resd_linears):
            net = self.actvn(self.resd_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        resd = self.resd_fc(net)
        resd = resd.transpose(1, 2)
        resd = self.c * torch.tanh(resd)

        return resd


class SDFNetwork(nn.Module):
    def __init__(self, layers=8):
        super(SDFNetwork, self).__init__()

        d_in = 3
        d_out = 257
        d_hidden = 256
        n_layers = layers

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        multires = 6
        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires,
                                                       input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        skip_in = [n_layers//2]
        bias = 0.5
        scale = 1
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs, batch):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def last_feature(self, inputs, batch):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        x = inputs
        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x

    def sdf(self, x, batch):
        return self.forward(x, batch)[:, :1]

    def gradient(self, x, batch):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x, batch)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class BetaNetwork(nn.Module):
    def __init__(self):
        super(BetaNetwork, self).__init__()
        init_val = 0.1
        self.register_parameter('beta', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        beta = self.beta
        # beta = torch.exp(self.beta).to(x)
        return beta


class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        self.color_latent = nn.Embedding(cfg.num_latent_code, 128)

        d_feature = 256
        mode = 'idr'
        d_in = 9
        d_out = 3
        d_hidden = 256
        n_layers = 4
        squeeze_out = True

        if not cfg.color_with_viewdir:
            mode = 'no_view_dir'

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden
                                     for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if self.mode != 'no_view_dir':
            multires_view = 4
            if multires_view > 0:
                embedview_fn, input_ch = embedder.get_embedder(multires_view)
                self.embedview_fn = embedview_fn
                dims[0] += (input_ch - 3)
        else:
            dims[0] = dims[0] - 3

        self.num_layers = len(dims)

        self.lin0 = nn.Linear(dims[0], d_hidden)
        self.lin1 = nn.Linear(d_hidden, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_hidden)
        self.lin3 = nn.Linear(d_hidden + 128, d_hidden)
        self.lin4 = nn.Linear(d_hidden, d_out)

        weight_norm = True
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors,
                latent_index):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat(
                [points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors],
                                        dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors],
                                        dim=-1)

        x = rendering_input

        net = self.relu(self.lin0(x))
        net = self.relu(self.lin1(net))
        net = self.relu(self.lin2(net))

        latent = self.color_latent(latent_index)
        latent = latent.expand(net.size(0), latent.size(1))
        features = torch.cat((net, latent), dim=1)

        net = self.relu(self.lin3(features))
        x = self.lin4(net)

        if self.squeeze_out:
            x = torch.sigmoid(x)

        return x
