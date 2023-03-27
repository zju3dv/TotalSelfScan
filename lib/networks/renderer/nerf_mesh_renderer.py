import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh
from lib.utils.blend_utils import *


class Renderer:
    def __init__(self, net):
        self.net = net

    def batchify_rays(self, wpts, decoder, chunk=1024 * 32):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[1]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = decoder(wpts[:, i:i + chunk])
            all_ret.append(ret[0, 0].detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape

        pts = pts.view(sh[0], -1, 3)

        # sampling points for blend weight training
        bw = pts_sample_blend_weights(pts, batch['tbw'], batch['tbounds'])
        tnorm = bw[:, 24]

        inside = tnorm < cfg.norm_th
        pts = pts[inside][None]

        tpose_human = self.net.tpose_human
        decoder = lambda x: tpose_human.calculate_alpha(x)

        alpha = self.batchify_rays(pts, decoder, 2048 * 64)

        inside = inside.detach().cpu().numpy()
        full_alpha = np.zeros(inside.shape)
        full_alpha[inside] = alpha

        cube = full_alpha.reshape(*sh[1:-1])
        cube = np.pad(cube, 10, mode='constant')
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        mesh = trimesh.Trimesh(vertices, triangles)

        ret = {'cube': cube, 'mesh': mesh}

        return ret
