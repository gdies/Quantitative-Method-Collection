# IMPORT MODULES
import numpy as np

# SUPPORT CLASSES
class OLS(object):
    """
    Implement linear regression, estimated with the OLS method
    """
    def __init__(self, fit_intercept:bool=True, check_singularity:bool=True):
        self.parameters = {'fit_intercept':fit_intercept, 'check_singularity':check_singularity}
        self.beta = None
        self.res = None
        self.covariance_matrix = None
        self.beta_std = None
        self.p_values = None

    def fit(self, X, y, compute_statistics:bool=False):
        """
        fit model, X is numpy.ndarray
        y is numpy.ndarray
        """
        if self.parameters['fit_intercept']:
            X = np.append(X, np.ones(shape=(X.shape[0],1), dtype=np.int8), axis=1)

        n, k = X.shape
        if self.parameters['check_singularity']:
            XX = np.transpose(X) @ X
            if np.linalg.matrix_rank(XX) == XX.shape[1]: # check if X'X is singular
                self.beta = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
            else:
                raise np.linalg.LinAlgError("X'X is singular (not invertible)")
        else:
            print(f"X shape: {X.shape}, rank: {np.linalg.matrix_rank(X)}") # show if there are multicollinearity issues
            self.beta = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

        if compute_statistics:
            self.res = y-self.predict(X, with_intercept=self.fit_intercept)
            self.covariance_matrix = self.get_covariance_matrix(X, self.beta, self.get_sigma_2(self.res, n, k))
            self.beta_std = np.sqrt(np.diag(self.covariance_matrix)) # standard deviation of beta
            self.p_values = self.get_p(z=self.beta/self.beta_std) # get p-values
            self.r_squared = self.get_R2(y=y, res=self.res)
            self.adjusted_r_squared = 1-(1-self.r_squared)*(n-1)/(n-k-1)

    def predict(self, X, add_intercept:bool=True):
        """
        Predict on data, return y_hat
        add_intercept : pick True if training set used intercept
        """
        try:
            X = X.to_numpy()
        except:
            pass
        if (self.parameters['fit_intercept'] and add_intercept):
            X = np.append(X, np.ones(shape=(X.shape[0], 1), dtype=np.int8), axis=1)
        n, k = X.shape # print(f'predict: data dimensions: X - {n}x{k}')
        return X @ self.beta

    def get_sigma_2(self, res, n:int, k:int):
        """
        If the model is fitted, get sigma^2 (estimate)
        """
        return np.dot(np.transpose(res), res)/(n-k)

    def get_covariance_matrix(self, X, beta, sigma_2):
        """
        inputs are all np.ndarray() -s
        """
        return sigma_2*np.linalg.inv((np.dot(np.transpose(X), X)))

    @staticmethod
    def get_p(z, sample_size:int=10000):
        """
        Calculate p-value, having z-statistics input array
        """
        normal_sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
        return np.array([np.where(normal_sample >= np.absolute(i), 1, 0).sum()/sample_size for i in z])

    def get_R2(self, y, res):
        """
        Compute and return R^2 - coefficient of determination
        """
        return 1 - np.sum(res**2)/np.sum(y**2)
