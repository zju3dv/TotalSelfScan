import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os
import cv2
from termcolor import colored


class Visualizer:
    def __init__(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

    def visualize_image(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        orig_H, orig_W = batch['orig_H'].item(), batch['orig_W'].item()
        mask_at_box = mask_at_box.reshape(H, W)
        img_pred = np.ones((H, W, 3)) * cfg.bg_color
        orig_img_pred = np.ones((orig_H+100, orig_W, 3)) * cfg.bg_color
        crop_bbox = batch['crop_bbox'][0].cpu()
        img_pred[mask_at_box] = rgb_pred
        orig_img_pred[crop_bbox[0,1]:crop_bbox[0,1]+H, crop_bbox[0,0]:crop_bbox[0,0]+W] = img_pred

        img_gt = np.zeros((H, W, 3))

        result_dir = os.path.join(cfg.result_dir, 'comparison_novel_view_{}'.format(batch['frame_index'].item()))
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (orig_img_pred[..., [2, 1, 0]] * 255))

    def visualize_normal(self, output, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)
        surf_mask = mask_at_box.copy()
        surf_mask[mask_at_box] = output['surf_mask'][0].detach().cpu().numpy()

        normal_map = np.zeros((H, W, 3))
        normal_map[surf_mask] = output['surf_normal'][
            output['surf_mask']].detach().cpu().numpy()

        normal_map[..., 1:] = normal_map[..., 1:] * -1
        norm = np.linalg.norm(normal_map, axis=2)
        norm[norm < 1e-8] = 1e-8
        normal_map = normal_map / norm[..., None]
        normal_map = (normal_map + 1) / 2

        plt.imshow(normal_map)
        plt.show()

    def visualize_acc(self, output, batch):
        acc_pred = output['acc_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        acc = np.zeros((H, W))
        acc[mask_at_box] = acc_pred

        plt.imshow(acc)
        plt.show()

        # acc_path = os.path.join(cfg.result_dir, 'acc')
        # i = batch['i'].item()
        # cam_ind = batch['cam_ind'].item()
        # acc_path = os.path.join(acc_path, '{:04d}_{:02d}.jpg'.format(i, cam_ind))
        # os.system('mkdir -p {}'.format(os.path.dirname(acc_path)))
        # plt.savefig(acc_path)

    def visualize_depth(self, output, batch):
        depth_pred = output['depth_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        depth = np.zeros((H, W))
        depth[mask_at_box] = depth_pred

        plt.imshow(depth)
        plt.show()

        # depth_path = os.path.join(cfg.result_dir, 'depth')
        # i = batch['i'].item()
        # cam_ind = batch['cam_ind'].item()
        # depth_path = os.path.join(depth_path, '{:04d}_{:02d}.jpg'.format(i, cam_ind))
        # os.system('mkdir -p {}'.format(os.path.dirname(depth_path)))
        # plt.savefig(depth_path)

    def visualize(self, output, batch):
        self.visualize_image(output, batch)
