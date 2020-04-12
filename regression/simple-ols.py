import numpy as np
import matplotlib.pyplot as plt 

# number of data points
N = 50

# construct N samples  following the line y = b1 +  b2x + u
# where the error term u ~ N(0, 0.5)
mu, sigma = 0, 0.5
b1, b2 = 1, 2
u = np.random.normal(mu, sigma, N)
X = np.arange(0, 5, 5/N)
#violate the requirement of homoscedasticity
u = u + X * X

Y = b1 + b2 * X + u

# To calculate the line of best fit solely from the 
# observed data tha we have
# We must Find the b1_hat, b2_hat that minimise 
# The Sum of Squared Errors

ybar = Y.mean()
xbar = X.mean()

b2_hat_numerator = X.dot(Y) - N * ybar * xbar
b2_hat_denom = X.dot(X) - N * xbar * xbar
b2_hat = b2_hat_numerator / b2_hat_denom

b1_hat = ybar - b2_hat * xbar

# Line of best fit
yhat = b1_hat + b2_hat * X

plt.scatter(X, Y)
plt.plot(X, yhat)
plt.show()
