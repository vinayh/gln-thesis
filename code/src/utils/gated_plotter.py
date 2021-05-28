import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class GatedPlotter():
    def __init__(self, X_all, y_all, add_ctx_fn):
        self.NX, self.NY = 120, 120
        self.Z_out_all = []
        self.plot, self.ax = plt.subplots()
        self.ax.scatter(X_all[:, 0], X_all[:, 1],
                        c=y_all, cmap='hot', marker='.',
                        linewidths=0.)
        self.x_min, self.x_max = self.ax.get_xlim()
        self.y_min, self.y_max = self.ax.get_ylim()
        xx = np.linspace(self.x_min, self.x_max, self.NX)
        # Y is from max to min (to plot correctly) instead of min to max
        yy = np.linspace(self.y_max, self.y_min, self.NY)
        XX, YY = np.meshgrid(xx, yy)
        self.xy = torch.tensor(np.stack([XX.ravel(), YY.ravel(),
                                         np.ones(XX.size)]).T,
                               dtype=torch.float)

        def add_to_plot_fn(Z_b):
            return self.ax.contour(XX, YY, Z_b.reshape(self.NX, self.NY),
                                   colors='k', levels=[0], alpha=0.5,
                                   linestyles=['-'])

        add_ctx_fn(self.xy, add_to_plot_fn)

    def save_data(self, forward_fn):
        Z_out = forward_fn(self.xy)
        self.Z_out_all.append(Z_out.reshape(self.NX, self.NY))

    def plot_step(self, Z_data):
        step_idx, Z_out = Z_data
        self.ax.imshow(Z_out.reshape(self.NX, self.NY),
                       vmin=Z_out.min(), vmax=Z_out.max(), cmap='viridis',
                       extent=[self.x_min, self.x_max,
                               self.y_min, self.y_max],
                       interpolation='none')
        # label = 'Step: {}'.format(step_idx)
        # self.ax.text(self.x_min+1, self.y_min+1, label,
        #              bbox={'facecolor': 'white', 'pad': 5})
        # # self.plot.suptitle(label)

    def save_animation(self, filename):
        anim = FuncAnimation(self.plot, self.plot_step,
                             frames=enumerate(self.Z_out_all))
        anim.save(filename, dpi=120, writer='imagemagick')
