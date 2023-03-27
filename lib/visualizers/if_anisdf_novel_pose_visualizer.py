import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os
from termcolor import colored


class Visualizer:
    def __init__(self):
        data_dir = 'data/novel_pose/{}'.format(cfg.exp_name)
        print(colored('the results are saved at {}'.format(data_dir),
                      'yellow'))

    def visualize(self, output, batch):
        pred_img = output['pred_img'][0].permute(1, 2, 0)
        pred_img = pred_img.detach().cpu().numpy()
        pred_msk = output['pred_msk'][0]
        pred_msk = pred_msk.detach().cpu().numpy()
        pred_img[pred_msk < 0.5] = 0
        pred_img = pred_img[..., [2, 1, 0]]

        orig_H, orig_W = batch['orig_H'].item(), batch['orig_W'].item()
        full_img_pred = np.zeros((orig_H, orig_W, 3))
        bbox = batch['crop_bbox'][0].detach().cpu().numpy()
        height, width = pred_img.shape[:2]
        full_img_pred[bbox[0, 1]:bbox[0, 1] + height,
                      bbox[0, 0]:bbox[0, 0] + width] = pred_img
        pred_img = full_img_pred

        view_index = batch['view_index'].item()
        img_root = 'data/novel_pose/{}/view{:04d}'.format(
            cfg.exp_name, view_index)
        os.system('mkdir -p {}'.format(img_root))

        frame_index = batch['frame_index'].item()
        cv2.imwrite(
            os.path.join(img_root, 'frame{:04d}.png'.format(frame_index)),
            pred_img * 255)
