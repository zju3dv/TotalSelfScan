import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os
from termcolor import colored


class Visualizer:
    def __init__(self):
        data_dir = 'data/novel_view/{}'.format(cfg.exp_name)
        print(colored('the results are saved at {}'.format(data_dir),
                      'yellow'))

    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        if cfg.white_bkgd:
            img_pred = img_pred + 1
        img_pred[mask_at_box] = rgb_pred
        img_pred = img_pred[..., [2, 1, 0]]
        mask_pred = np.zeros((H, W))
        mask_pred[mask_at_box] = output['pred_mask'][0].detach().cpu().numpy()
        # import matplotlib.pylab as plt;plt.figure();plt.imshow(mask_pred);plt.show()
        # import matplotlib.pylab as plt;plt.figure();plt.imshow(img_pred);plt.show()


        depth_pred = np.zeros((H, W))
        depth_pred[mask_at_box] = output['depth_map'][0].detach().cpu().numpy()


        if cfg.vis_train_view:
            if cfg.train_bgfg:
                img_root = 'data/train_view/{}_{}'.format(
                    cfg.exp_name, cfg.train_view_cfg.vis_type)
            else:
                img_root = 'data/train_view/{}'.format(
                    cfg.exp_name)
            os.system('mkdir -p {}'.format(img_root))
            if 'frame_index' in batch.keys():
                index = batch['frame_index'].item()
                img_name = '{:04d}.png'.format(index)
                mask_name = 'mask_{:04d}.png'.format(index)
            else:
                index = batch['meta']['frame_index'][0]
                cam_id = batch['cam_ind'].cpu().item()
                img_name = 'cam_{}_{}.png'.format(cam_id, index)

        else:
            img_root = 'data/novel_view/{}/frame_{:04d}'.format(
                cfg.exp_name, batch['frame_index'].item())
            os.system('mkdir -p {}'.format(img_root))
            index = batch['view_index'].item()
            img_name = '{:04d}.png'.format(index)

        cv2.imwrite(os.path.join(img_root, img_name), img_pred * 255)
        cv2.imwrite(os.path.join(img_root, mask_name), mask_pred * 255)

