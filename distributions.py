# A collection of cdf-s and pdf-s

# DEPENDENCIES
import numpy as np
from scipy import special

class Normal(object):
    """
    Implement normal distribution
    """
    def __init__(self, mu:float = 0, sigma:float = 1):
        self.mu = mu
        self.sigma = sigma

    def cdf(self, x):
        """
        Evaluate CDF at x
        """
        return (1 + special.erf((x - self.mu) / (self.sigma * np.sqrt(2)))) / 2

    def pdf(self, x):
        """
        Evaluate PDF at x
        """
        return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp((-1)/2 * ((x - self.mu) / self.sigma)**2)
