import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os
import cv2


class Visualizer:
    def visualize(self, output, batch):
        img_pred = output['rgb'][0].permute(1, 2, 0).detach().cpu().numpy()
        img_gt = batch['img'][0].permute(1, 2, 0).detach().cpu().numpy()

        result_dir = os.path.join('data/result/nt', cfg.exp_name)
        i = batch['i'].item()
        i = i + cfg.begin_i
        cam_ind = batch['cam_ind'].item()
        frame_dir = os.path.join(result_dir, 'frame_{}'.format(i))
        pred_img_path = os.path.join(frame_dir,
                                     'pred_{}.jpg'.format(cam_ind + 1))
        os.system('mkdir -p {}'.format(os.path.dirname(pred_img_path)))
        # img_pred = (img_pred * 255)[..., [2, 1, 0]]
        # cv2.imwrite(pred_img_path, img_pred)

        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img_pred)
        ax2.imshow(img_gt)
        plt.show()
