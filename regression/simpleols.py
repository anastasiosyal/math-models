import numpy as np

class simpleols:

    def estimate(self, b1 = 1, b2 = 2, N=50, error_sigma=1, maxX=5):
        """
            Creates a sample of Size N data points sampled from 
            Y = b1 + b2*X + u
            where u ~ N (0, error_sigma) around the line
            and x: N equally spaced samples between zero and maxX

            From the sample, the line of best fit is identified
            that minimises that SSE by estimating b1_hat, b2_hat

            Y_hat = b1_hat + b2_hat*X 
            
            returns: 
                X: the x axis values
                Y: the true values for X
                Y_hat: the estimated line of best fit
                b1_hat: OLS estimator for b1
                b2_hat: OLS estimator for b2

        """

        # construct N samples  following the line y = b1 + b2x + u
        # where the error term u ~ N(0, 0.5)
        u = np.random.normal(0, error_sigma, N)
        X = np.arange(0, maxX, maxX/N)

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
        Y_hat = b1_hat + b2_hat * X

        return (X, Y, Y_hat, b1_hat, b2_hat)