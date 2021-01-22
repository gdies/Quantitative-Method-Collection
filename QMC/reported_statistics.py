import numpy as np


def get_p(theta:np.ndarray, covariance_matrix:np.ndarray, sample_size:int=10000):
    """
    Calculate p-values for given parameters and their covariance matrix
    Assume asymptotic normality
    """
    std_theta = np.sqrt(np.diag(covariance_matrix)) # standard deviation of beta
    z = theta / std_theta # get z-statistics
    normal_sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    return np.array([np.where(normal_sample >= np.absolute(i), 1, 0).sum()/sample_size for i in z]) # p-values for each element in theta
