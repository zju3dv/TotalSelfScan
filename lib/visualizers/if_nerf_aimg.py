import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os
from termcolor import colored


class Visualizer:
    def __init__(self):
        data_dir = 'data/result/if_nerf/{}'.format(cfg.exp_name)
        print(colored('the results are saved at {}'.format(data_dir),
                      'yellow'))

    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        # print('mse: {}'.format(np.mean((rgb_pred - rgb_gt) ** 2)))

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred

        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        result_dir = os.path.join('data/result/if_nerf', cfg.exp_name)
        i = batch['real_i'].item()
        i = i + cfg.begin_i
        cam_ind = batch['cam_ind'].item()
        frame_dir = os.path.join(result_dir, 'cam_{}'.format(cam_ind))
        pred_img_path = os.path.join(frame_dir,
                                     'pred_{}.jpg'.format(i))
        os.system('mkdir -p {}'.format(os.path.dirname(pred_img_path)))
        img_pred = (img_pred * 255)[..., [2, 1, 0]]
        cv2.imwrite(pred_img_path, img_pred)

        gt_img_path = os.path.join(frame_dir, 'gt_{}.jpg'.format(i))
        os.system('mkdir -p {}'.format(os.path.dirname(gt_img_path)))
        img_gt = (img_gt * 255)[..., [2, 1, 0]]
        cv2.imwrite(gt_img_path, img_gt)

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_pred)
        # ax2.imshow(img_gt)
        # plt.show()
