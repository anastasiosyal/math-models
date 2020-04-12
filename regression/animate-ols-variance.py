import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import Animation
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import math


from simpleols import simpleols

numRegressions = 200
ols = simpleols()

fig = plt.figure(tight_layout=True)
gs = GridSpec(2, 2, figure=fig)
ax = fig.add_subplot(gs[0, :])
axb1 = fig.add_subplot(gs[1, 0])
axb2 = fig.add_subplot(gs[1, 1])
for axis, i  in [(axb1, 0), (axb2, 1)]:
    axis.set_title('Histogram of $\\hat{\\beta}_' + str(i + 1)+ r'$ estimator')
    axis.xaxis.set_animated(True)
    axis.yaxis.set_animated(True)

(X, Y, Y_hat, b1_hat, b2_hat) = ols.estimate()
b1_hat_var = 1 / X.dot(X)

b1_hats = []
b2_hats = []

scat = ax.scatter(X, Y)
lines = []
title = ax.text(.5,.9,'centered title',
    horizontalalignment='center',
    transform=ax.transAxes,
    backgroundcolor='white')    

def init():
    for i in range(numRegressions):
        lobj = ax.plot([],[],'r-', alpha=0.1)[0]
        lines.append(lobj)
    return lines

def update(frame):

    (X, Y, Y_hat, b1_hat, b2_hat) = ols.estimate()
    b1_hats.append(b1_hat)
    b2_hats.append(b2_hat)
    lines[frame + 1].set_data(X, Y_hat)
    scatter_offsets = np.hstack((X[:,np.newaxis],Y[:,np.newaxis]))
    scat.set_offsets(scatter_offsets)
    title.set_text(f'Regression #{frame + 1}')
    if frame > 30:
        ani.event_source.interval = 50
    elif frame > 10:
        ani.event_source.interval = 80
    else:
        ani.event_source.interval = 400

    # plot histograms of beta values
    if (frame > 1): 
         axb1.clear()
         axb1.hist(b1_hats,bins=min(frame,20),normed=True)
         axb2.clear()
         axb2.hist(b2_hats,bins=min(frame,20),normed=True)

    return [scat, *lines, title, axb1, axb2, axb1.xaxis, axb1.yaxis, axb2.xaxis, axb2.yaxis]


# need to use this to update the axis
def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

# MONKEY PATCH!!
Animation._blit_draw = _blit_draw

ani = FuncAnimation(fig, update, frames=np.arange(numRegressions),
                    init_func=init, blit=True, interval=1, repeat=False)
plt.show()
