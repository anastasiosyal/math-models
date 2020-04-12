import numpy as np
import matplotlib.pyplot as plt 
from simpleols import simpleols

ols = simpleols()
(X, Y, Y_hat, b1_hat, b2_hat) = ols.estimate()

plt.scatter(X, Y)
plt.plot(X, Y_hat, 'r-', alpha=0.2)
plt.show()
