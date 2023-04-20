# import open3d as o3d
from pytorch3d.renderer.cameras import PerspectiveCameras, OrthographicCameras
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)
import pytorch3d.structures as struct
from tqdm import tqdm
import os
import glob
import numpy as np
import trimesh
import torch
import cv2
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.networks.renderer import make_renderer
from lib.utils.blend_utils import pts_sample_blend_weights
from lib.config import cfg
# from lib.utils.vis_utils import get_colored_pc
from lib.utils.blend_utils import *
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.datasets import make_data_loader
import matplotlib.pyplot as plt
import time
import imageio


def set_pytorch3d_intrinsic_matrix(K, H, W):
    img_size = min(H, W)
    fx = -K[0, 0] * 2.0 / img_size
    fy = -K[1, 1] * 2.0 / img_size
    px = -(K[0, 2] - W / 2.0) * 2.0 / img_size
    py = -(K[1, 2] - H / 2.0) * 2.0 / img_size
    K = [
        [fx, 0, px, 0],
        [0, fy, py, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    K = np.array(K)
    return K


def padding_bbox(bbox, img):
    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    # a magic number of pytorch3d
    ratio = 1.9

    if height / width > ratio:
        min_size = int(height / ratio)
        if width < min_size:
            padding = (min_size - width) // 2
            bbox[0, 0] = bbox[0, 0] - padding
            bbox[1, 0] = bbox[1, 0] + padding

    if width / height > ratio:
        min_size = int(width / ratio)
        if height < min_size:
            padding = (min_size - height) // 2
            bbox[0, 1] = bbox[0, 1] - padding
            bbox[1, 1] = bbox[1, 1] + padding

    h, w = img.shape[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], a_min=0, a_max=w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], a_min=0, a_max=h - 1)

    return bbox


def revise_K_msk(K, msk):
    x, y, w, h = cv2.boundingRect(msk)
    bbox = np.array([[x, y], [x + w, y + h]])
    bbox = padding_bbox(bbox, msk)

    # calculate the shape
    shape = (bbox[1, 1] - bbox[0, 1], bbox[1, 0] - bbox[0, 0])
    x = 8
    height = int((shape[0] | (x - 1)) + 1)
    width = int((shape[1] | (x - 1)) + 1)

    height = int(min(msk.shape[0] - bbox[0, 1], height))
    width = int(min(msk.shape[1] - bbox[0, 0], width))

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - bbox[0, 0]
    K[1, 2] = K[1, 2] - bbox[0, 1]
    K = K.astype(np.float32)

    return K, bbox, height, width


def raw2outputs(raw, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = raw[..., :-1]  # [N_rays, N_samples, 3]
    alpha = raw[..., -1]

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, acc_map, weights


class Renderer:
    def __init__(self):
        data_root = cfg.test_dataset.data_root
        ann_file = cfg.test_dataset.ann_file

        # network
        network = make_network(cfg).cuda()
        load_network(network,
                     cfg.trained_model_dir,
                     resume=cfg.resume,
                     epoch=cfg.test.epoch,
                     strict=False)
        self.net = network.train()

        # dataset
        cfg.test_dataset_module =  'lib.datasets.h36m.tpose_animation'
        cfg.test_dataset_path = 'lib/datasets/h36m/tpose_animation.py'
        self.data_loader = make_data_loader(cfg, is_train=False)
        dataset = self.data_loader.dataset
        self.dataset = dataset

        # rasterizer
        cam_ind = 3
        self.K = np.array(dataset.cams['K'][cam_ind])
        self.K[:2] = self.K[:2] * cfg.ratio
        H, W = imageio.imread(os.path.join(data_root,
                                           dataset.ims[0])).shape[:2]
        H, W = int(H * cfg.ratio), int(W * cfg.ratio)
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(self.K, H, W)
        self.R = np.array(dataset.cams['R'][cam_ind])
        self.T = np.array(dataset.cams['T'][cam_ind]) / 1000.
        self.H = H
        self.W = W

        animation_root = os.path.join('data/animation', cfg.exp_name)
        self.render_root = os.path.join(animation_root, 'render')
        os.system('mkdir -p {}'.format(self.render_root))

        tpose_dir = 'data/animation/{}/mesh.npy'.format(cfg.exp_name)
        tpose_mesh = np.load(tpose_dir, allow_pickle=True).item()
        self.tvertex = torch.from_numpy(tpose_mesh['vertex']).cuda()
        triangle = tpose_mesh['triangle'].astype(np.int32)
        self.tface = torch.from_numpy(triangle).cuda()
        self.bw = torch.from_numpy(tpose_mesh['blend_weight'][:24]).cuda()

        ray_o, ray_d = if_nerf_dutils.get_rays(self.H, self.W, self.K, self.R,
                                               self.T)
        self.ray_o_map = torch.FloatTensor(ray_o.copy()).cuda()
        self.ray_d_map = torch.FloatTensor(ray_d.copy()).cuda()

    def image_rendering(self, ret, mask, decoder):
        H, W = mask.shape
        pixel_rgb_map = ret['rgb_map']
        n_channel = pixel_rgb_map.shape[1]
        rgb_map = torch.zeros([1, H, W, n_channel]).to(pixel_rgb_map)
        rgb_map[mask[None] == 1] = pixel_rgb_map

        rgb_map = rgb_map.permute(0, 3, 1, 2).contiguous()
        latent_index = torch.tensor([0]).to(rgb_map.device)
        raw = decoder(rgb_map, {'latent_index': latent_index})
        pred_img = raw[:, :3]
        pred_msk = raw[:, 3]
        ret = {'pred_img': pred_img, 'pred_msk': pred_msk}

        return ret

    def render(self):
        dataset = self.dataset

        tvertex = self.tvertex[None].float()
        bw = self.bw[None].float()
        triangle = self.tface[None]

        RT = np.concatenate([self.R, self.T], axis=1)

        for batch in tqdm(self.data_loader):
            for k in ['big_A', 'A', 'R', 'Th']:
                batch[k] = batch[k].cuda()

            # crop image using bounds
            wbounds = batch['wbounds'][0].detach().cpu().numpy()
            bound_mask = if_nerf_dutils.get_bound_2d_mask(
                wbounds, self.K, RT, self.H, self.W)
            K, bbox, height, width = revise_K_msk(self.K, bound_mask)

            pytorch3d_K = set_pytorch3d_intrinsic_matrix(K, height, width)
            cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K[None].astype(
                                             np.float32),
                                         R=self.R.T[None].astype(np.float32),
                                         T=self.T.T.astype(np.float32))
            raster_settings = RasterizationSettings(image_size=(height, width),
                                                    blur_radius=0.0,
                                                    faces_per_pixel=1,
                                                    bin_size=None)
            rasterizer = MeshRasterizer(cameras=cameras,
                                        raster_settings=raster_settings)

            # animate mesh
            ppose = pose_points_to_tpose_points(tvertex, bw, batch['big_A'])
            pvertex_i = tpose_points_to_pose_points(ppose, bw, batch['A'])
            vertex = pose_points_to_world_points(pvertex_i, batch['R'],
                                                 batch['Th'])

            # rasterize mesh
            torch.cuda.synchronize()
            now = time.time()
            ppose = struct.Meshes(verts=vertex, faces=triangle)
            fragments = rasterizer(ppose)

            face_idx_map = fragments.pix_to_face[0, ..., 0]
            bary_coords_map = fragments.bary_coords[0, :, :, 0]
            torch.cuda.synchronize()
            print('rasterization: {}'.format(time.time() - now))

            mask = face_idx_map >= 0
            pixel_face_idx = face_idx_map[mask]
            pixel_bary_coord = bary_coords_map[mask]

            torch.cuda.synchronize()
            now = time.time()
            # get 3d points
            pixel_vertex_idx = triangle[0][pixel_face_idx].long()
            pixel_face_vertex = vertex[0][pixel_vertex_idx]
            pixel_vertex = torch.sum(pixel_face_vertex *
                                     pixel_bary_coord[..., None],
                                     dim=1)

            # get blend weights of 3d points
            pixel_face_bw = bw[0].T[pixel_vertex_idx]
            pixel_bw = torch.sum(pixel_face_bw * pixel_bary_coord[..., None],
                                 dim=1)

            # sample 3d points along ray
            ray_d_map = self.ray_d_map[bbox[0, 1]:bbox[0, 1] + height,
                                       bbox[0, 0]:bbox[0, 0] + width]
            pixel_ray_d = ray_d_map[mask]
            n_sample = cfg.N_samples // 2 * 2 + 1
            z_interval = 0.04
            t_vals = torch.linspace(-z_interval, z_interval,
                                    steps=n_sample).to(pixel_ray_d)
            wpts = pixel_vertex[:,
                                None] + pixel_ray_d[:, None] * t_vals[:, None]
            n_pixel, n_sample = wpts.shape[:2]
            wpts_bw = pixel_bw[:, None].expand(-1, n_sample, -1).contiguous()
            wpts_ray_d = pixel_ray_d[:, None].expand(-1, n_sample,
                                                     -1).contiguous()

            wpts = wpts.view(1, n_pixel * n_sample, 3)
            wpts_bw = wpts_bw.view(1, n_pixel * n_sample, 24).permute(0, 2, 1)
            wpts_ray_d = wpts_ray_d.view(1, n_pixel * n_sample, 3)

            # transform points from the world space to the pose space
            pose_pts = world_points_to_pose_points(wpts, batch['R'],
                                                   batch['Th'])

            # transform points from the pose space to the bigpose space
            tpose = pose_points_to_tpose_points(pose_pts, wpts_bw, batch['A'])
            bigpose = tpose_points_to_pose_points(tpose, wpts_bw,
                                                  batch['big_A'])
            torch.cuda.synchronize()
            print('sample points: {}'.format(time.time() - now))

            n_point = n_pixel * n_sample
            chunk = n_sample * 2048

            torch.cuda.synchronize()
            now = time.time()
            latent_index = torch.tensor([0]).to(bigpose.device)
            ret_list = []
            with torch.no_grad():
                for i in range(0, n_point, chunk):
                    bigpose_chunk = bigpose[0, i:i + chunk]
                    ray_d_chunk = wpts_ray_d[0, i:i + chunk]
                    ret = self.net.tpose_human(bigpose_chunk, ray_d_chunk,
                                               None,
                                               {'latent_index': latent_index})

                    raw = ret['raw'].reshape(-1, n_sample, ret['raw'].size(1))
                    rgb_map, acc_map, weights = raw2outputs(
                        raw, cfg.white_bkgd)
                    ret = {
                        'rgb_map': rgb_map,
                        'acc_map': acc_map,
                    }
                    ret_list.append(ret)

            keys = ret_list[0].keys()
            ret = {k: torch.cat([r[k] for r in ret_list], dim=0) for k in keys}

            if ret['rgb_map'].shape[1] > 3:
                ret = self.image_rendering(ret, mask, self.net.image_renderer)
                img_pred = ret['pred_img'][0].permute(1, 2, 0)
                img_pred = img_pred.detach().cpu().numpy()
                msk_pred = ret['pred_msk'][0]
                msk_pred = msk_pred.detach().cpu().numpy()
                img_pred[msk_pred < 0.5] = 0
            else:
                mask = mask.detach().cpu().numpy()
                H, W = mask.shape[:2]
                img_pred = np.zeros((H, W, 3))
                img_pred[mask] = ret['rgb_map'].detach().cpu().numpy()

            torch.cuda.synchronize()
            print('rendering: {}'.format(time.time() - now))

            full_img_pred = np.zeros((self.H, self.W, 3))
            full_img_pred[bbox[0, 1]:bbox[0, 1] + height,
                          bbox[0, 0]:bbox[0, 0] + width] = img_pred

            frame_index = batch['frame_index'].item()
            img_path = os.path.join(self.render_root,
                                    '{:04d}.png'.format(frame_index))
            cv2.imwrite(img_path, full_img_pred[..., [2, 1, 0]] * 255)
            # plt.imshow(full_img_pred)
            # plt.show()
