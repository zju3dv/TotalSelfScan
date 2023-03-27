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

        result_dir = os.path.join('data/result/nt_render', cfg.exp_name)
        i = batch['i'].item()
        cam_ind = batch['index'].item()
        pred_img_path = os.path.join(result_dir,
                                     '{}/{}.jpg'.format(i, cam_ind))
        os.system('mkdir -p {}'.format(os.path.dirname(pred_img_path)))
        img_pred = (img_pred * 255)[..., [2, 1, 0]]
        cv2.imwrite(pred_img_path, img_pred)
