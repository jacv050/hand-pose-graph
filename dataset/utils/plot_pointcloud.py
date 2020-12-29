import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Plot:
  def __init__(self):
    self.fig = plt.figure()
    self.ax = self.fig.add_subplot(111, projection='3d')

    self.ax.set_xlabel('X Label')
    self.ax.set_ylabel('Y Label')
    self.ax.set_zlabel('Z Label')

  def cloud(self, cloud, minmax):
    x, y, z = cloud[:,0], cloud[:,1], cloud[:,2]
    self.ax.scatter(x, y, z, c='r', marker='.')
    #Draw cube
    x_min_max = minmax[0]
    y_min_max = minmax[1]
    z_min_max = minmax[2]
    #min
    P1=[x_min_max[0], y_min_max[0], z_min_max[0]]
    P8=[x_min_max[1], y_min_max[1], z_min_max[1]]
    P2=[P8[0], P1[1], P1[2]]
    P3=[P8[0], P1[1], P8[2]]
    P4=[P1[0], P1[1], P8[2]]
    P5=[P8[0], P8[1], P1[2]]
    P6=[P1[0], P8[1], P1[2]]
    P7=[P1[0], P8[1], P8[2]]
    self.ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]])
    self.ax.plot([P2[0], P3[0]], [P2[1], P3[1]], [P2[2], P3[2]])
    self.ax.plot([P3[0], P4[0]], [P3[1], P4[1]], [P3[2], P4[2]])
    self.ax.plot([P1[0], P4[0]], [P1[1], P4[1]], [P1[2], P4[2]])
    self.ax.plot([P2[0], P5[0]], [P2[1], P5[1]], [P2[2], P5[2]])
    self.ax.plot([P5[0], P8[0]], [P5[1], P8[1]], [P5[2], P8[2]])
    self.ax.plot([P3[0], P8[0]], [P3[1], P8[1]], [P3[2], P8[2]])
    self.ax.plot([P1[0], P6[0]], [P1[1], P6[1]], [P1[2], P6[2]])
    self.ax.plot([P7[0], P6[0]], [P7[1], P6[1]], [P7[2], P6[2]])
    self.ax.plot([P7[0], P4[0]], [P7[1], P4[1]], [P7[2], P4[2]])
    self.ax.plot([P7[0], P8[0]], [P7[1], P8[1]], [P7[2], P8[2]])
    self.ax.plot([P5[0], P6[0]], [P5[1], P6[1]], [P5[2], P6[2]])

  def show(self):
    plt.show()
    plt.waitforbuttonpress()
