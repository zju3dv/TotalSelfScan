# @Author  : Junting Dong
# @Mail    : jtdong@zju.edu.cn

import numpy as np
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

class Debugger(object):
    def __init__(self):
        self.plt = plt

    def init_fig3d(self):
        self.fig = self.plt.figure()
        self.ax = self.fig.add_subplot((111), projection='3d')
        self.ax.set_xlabel('z')
        self.ax.set_ylabel('x')
        self.ax.set_zlabel('y')
    def init_fig2d(self):
        self.fig = self.plt.figure()
        self.ax = self.fig.add_subplot(111)

    def addShow(self, point, c='b', lim=True):
        # point: N*3
        self.init_fig3d()
        self.addPoint3D(point, c)
        if lim:
            self.setLim(point)
        self.show()

    def showL2D(self, imgs):
        n_img = len(imgs)
        self.fig = self.plt.figure()
        for idx, img in enumerate(imgs):
            self.ax = self.fig.add_subplot(1, n_img, idx+1)
            self.ax.imshow(img)
        self.show()


    def setLim(self, point):
        x_l = point[:, 0].max() - point[:, 0].min()
        y_l = point[:, 1].max() - point[:, 1].min()
        z_l = point[:, 2].max() - point[:, 2].min()
        x_c = (point[:, 0].max() + point[:, 0].min()) / 2
        y_c = (point[:, 1].max() + point[:, 1].min()) / 2
        z_c = (point[:, 2].max() + point[:, 2].min()) / 2
        length = np.max([x_l, y_l, z_l])
        self.ax.set_xlim3d(x_c - length, x_c + length)
        self.ax.set_ylim3d(y_c - length, y_c + length)
        self.ax.set_zlim3d(z_c - length, z_c + length)

    def addPoint3D(self, point, c='b'):
        self.ax.scatter(point[:, 0], point[:, 1], point[:, 2], c)

    def showL3D(self, point, c='b', lim=True):
        self.fig = self.plt.figure()
        self.ax = self.fig.add_subplot((111), projection='3d')

        for i in range(len(point)):
            self.ax.scatter(point[i][:, 0], point[i][:, 1], point[i][:, 2], c)
        # self.ax.plot(point[:, 0], point[:, 1], point[:, 2], c)
        if lim:
            self.setLim(np.concatenate(point, axis=0))
        self.plt.show()

    def addRepro(self, img, point, K, Rc=np.eye(3), Tc=np.zeros([3,1])):
        self.init_fig2d()
        point_c = (point @ Rc.T + Tc.T) @ K.T
        point_c = point_c[:, :2] / point_c[:, 2:]
        self.ax.imshow(img)
        self.ax.plot(point_c[:, 0], point_c[:, 1], 'r*')
        self.show()


    def show(self):

        self.plt.show()

dbg = Debugger()
