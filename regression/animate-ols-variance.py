import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import Animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import math

from simpleols import simpleols

numRegressions = 150
ols = simpleols()

fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
ax = fig.add_subplot(gs[0, :])
ax.set_title("Overlapped regression lines")
axb = fig.add_subplot(gs[1, :])
axb.set_title(r'Histograms of $\beta estimators: \hat{\beta}_1 (blue), \hat{\beta}_1 (orange)$ ')
axb.hist([],bins=20,density=False)

(X, Y, Y_hat, b1_hat, b2_hat) = ols.estimate()
b1_hat_var = 1 / X.dot(X)

#simulation data
Xs, Ys, Y_hats = [], [], []
b1_hats = []
b2_hats = []

for i in range(numRegressions):
    (X, Y, Y_hat, b1_hat, b2_hat) = ols.estimate()
    Xs.append(X)
    Ys.append(Y)
    Y_hats.append(Y_hat)
    b1_hats.append(b1_hat)
    b2_hats.append(b2_hat)


scat = ax.scatter(X, Y)
lines = []
title = ax.text(.5,.8,'centered title',
    horizontalalignment='center',
    transform=ax.transAxes,
    backgroundcolor='white')    

for i in range(numRegressions):
    lobj = ax.plot([],[],'r-', alpha=0.1)[0]
    lines.append(lobj)


def update(frame):
    lines[frame].set_data(Xs[frame], Y_hats[frame])
    scatter_offsets = np.hstack((Xs[frame][:,np.newaxis],Ys[frame][:,np.newaxis]))
    scat.set_offsets(scatter_offsets)
    title.set_text(f'Regression #{frame + 1}')
    if frame > 30:
        ani.event_source.interval = 5
    elif frame > 10:
        ani.event_source.interval = 20
    else:
        ani.event_source.interval = 400

    # plot histograms of beta values
    axb.clear()
    axb.set_title(r'Histograms of $\beta$ estimators: $\hat{\beta}_1 (blue), \hat{\beta}_1 (orange)$ ')
    axb.hist(b1_hats[:frame],bins=20,density=False,alpha=0.7)
    axb.hist(b2_hats[:frame],bins=20,density=False,alpha=0.7)

    # return [scat, *lines, title, axb, axb2, axb.xaxis, axb.yaxis, axb2.xaxis, axb2.yaxis]


ani = FuncAnimation(fig, update, frames=np.arange(numRegressions),
                    blit=False, interval=50, repeat=False)
writer = PillowWriter(fps=20)
ani.save('anim.gif', writer=writer)
plt.show()
