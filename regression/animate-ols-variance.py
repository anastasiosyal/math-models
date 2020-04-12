import numpy as np
import matplotlib.pyplot as plt 
from simpleols import simpleols
from matplotlib.animation import FuncAnimation

numRegressions = 200
ols = simpleols()
(X, Y, Y_hat, b1_hat, b2_hat) = ols.estimate()
fig, ax = plt.subplots()
scat = plt.scatter(X, Y)
lines = [scat]
intervals = np.ones(numRegressions)


def init():
    for i in range(numRegressions):
        lobj = ax.plot([],[],'r-', alpha=0.04)[0]
        lines.append(lobj)
    return lines

def update(frame):
    (X, Y, Y_hat, b1_hat, b2_hat) = ols.estimate()
    lines[frame + 1].set_data(X, Y_hat)
    scatter_offsets = np.hstack((X[:,np.newaxis],Y[:,np.newaxis]))
    scat.set_offsets(scatter_offsets)
    ani.event_source.interval = numRegressions / 2 * np.exp(-frame/numRegressions) 
    return lines

ani = FuncAnimation(fig, update, frames=np.arange(numRegressions),
                    init_func=init, blit=True, interval=1, repeat=False)
plt.show()
