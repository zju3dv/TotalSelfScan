import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os
import cv2
from termcolor import colored


class Visualizer:
    def __init__(self):
        self.result_dir = 'data/view_dirs/{}'.format(cfg.exp_name)
        os.makedirs(self.result_dir, exist_ok=True)
        print(
            colored('the results are saved at {}'.format(self.result_dir),
                    'yellow'))

    def visualize_image(self, output, batch):
        if batch['last_frame'] == 1:
            view_dirs = output['view_dirs_statics']
            for part in view_dirs.keys():
                if len(view_dirs[part]) == 0:
                    continue
                # num_pts_max = max(view_dirs[part].items(), key=lambda kv: kv[1].shape[0])[1].shape[0]
                # view_dirs_part = np.zeros([output['n_face'], num_pts_max, 3])
                for face_idx in view_dirs[part].keys():
                    view_dirs[part][face_idx] = view_dirs[part][face_idx].cpu().numpy()
            np.save('{}/view_dirs.npy'.format(self.result_dir), view_dirs)


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
