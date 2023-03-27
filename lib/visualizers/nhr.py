import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os
import cv2


class Visualizer:
    def visualize(self, output, batch):
        img_pred = output['rgb'][0].permute(1, 2, 0).detach().cpu().numpy()
        mask = output['mask'][0, 0].detach().cpu().numpy()
        img_pred[mask < 0.5] = 0

        img_gt = batch['img'][0].permute(1, 2, 0).detach().cpu().numpy()

        result_dir = os.path.join('data/result/nhr', cfg.exp_name)
        i = batch['i'].item()
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
