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
        pred_img = output['pred_img'][0].permute(1, 2, 0)
        pred_img = pred_img.detach().cpu().numpy()
        pred_msk = output['pred_msk'][0]
        pred_msk = pred_msk.detach().cpu().numpy()
        pred_img[pred_msk < 0.5] = 0
        pred_img = pred_img[..., [2, 1, 0]]

        frame_index = batch['frame_index'].item()
        img_root = 'data/novel_view/{}/frame{:04d}'.format(
            cfg.exp_name, frame_index)
        os.system('mkdir -p {}'.format(img_root))

        view_index = batch['view_index'].item()
        cv2.imwrite(
            os.path.join(img_root, 'view{:04d}.png'.format(view_index)),
            pred_img * 255)
