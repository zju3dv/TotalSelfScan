import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os


class Visualizer:
    def visualize(self, output, batch):
        pred_img = output['pred_img'][0].permute(1, 2, 0)
        pred_img = pred_img.detach().cpu().numpy()
        pred_msk = output['pred_msk'][0]
        pred_msk = pred_msk.detach().cpu().numpy()
        pred_img[pred_msk < 0.5] = 0

        img = batch['img'][0].permute(1, 2, 0)
        img = img.detach().cpu().numpy()

        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(pred_img)
        ax2.imshow(img)
        ax3.imshow(pred_msk)
        plt.show()
