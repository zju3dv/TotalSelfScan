import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg


class Visualizer:
    def visualize(self, output, batch):
        bg = output.detach().cpu().numpy()
        row1 = np.concatenate([bg[0], bg[1]], axis=1)
        row2 = np.concatenate([bg[2], bg[3]], axis=1)
        bg = np.concatenate([row1, row2], axis=0)
        plt.imshow(bg)
        plt.show()
