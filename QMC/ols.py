# IMPORT MODULES
import numpy as np

# SUPPORT CLASSES
class OLS(object):
    """
    This class implements linear regression, estimated with the OLS method

    :param fit_intercept: If True, a constant term is added to the linear model.
    :type fit_intercept: bool, optional, defaults to True
    :param check_singularity: If True, checks the rank of X'X. Where X is represents the data.
    :type check_singularity: bool, optional, defaults to True
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
        Fit/Estimate model using data X.

        :param X: Input data / Independent Variables, a matrix with (n x k) dimensions.
        :type X: numpy.ndarray
        :param y: Dependent / output variable, a vector of (n x 1) dimensions.
        :type y: numpy.ndarray
        :param compute_statistics: If True, standard errors and p-values are computed for the point estimates as well as R-squared and adjusted R-squared values for the model.
        :type compute_statistics: bool, optional, defaults to False

        :raises LinAlgError: If the X'X matrix is singular, it is not invertible, thus the estimation breaks down.

        :return: Only updates the OLS object
        :rtype: None
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
        Predict output given some input data.

        :param X: Input data (test set) with the same column dimensions as the data used to fit the model (training set).
        :type X: numpy.ndarray
        :param add_intercept: If True, an intercept/constant term is added to the data. Pick only if it was uded for the training data as well.
        :type add_intercept: bool, optional, default to True

        :return: y_hat, prediction of output/dependent variable
        :rtype: numpy.ndarray
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
        If the model is fitted, get sigma^2 (estimate), that is the variance of the residual terms.

        :param res: The (n x 1) vector of the residual terms of a fitted model.
        :type res: numpy.ndarray
        :param n: The first dimension of the fitted data, that is the number of rows.
        :type n: int
        :param k: The second dimension of the fitted data, that is the number of columns / independent variables (including the intercept term).
        :type k: int

        :return: sigma^2 (1 x 1), the variance of the residual terms
        :rtype: numpy.ndarray
        """
        return np.dot(np.transpose(res), res)/(n-k)

    def get_covariance_matrix(self, X, beta, sigma_2):
        """
        This function computes the covariance matrix of the estimated model parameters.

        :param X: Input data (test set) with the same column dimensions as the data used to fit the model (training set).
        :type X: numpy.ndarray
        :param beta: Parameter vector (unused).
        :type beta: numpy.ndarray
        :param sigma_2: The variance of the residual terms.
        :type sigma_2: numpy.ndarray

        :return: The (k x k) covariance matrix of the estimated model parameters.
        :rtype: numpy.ndarray
        """
        return sigma_2*np.linalg.inv((np.dot(np.transpose(X), X)))

    @staticmethod
    def get_p(z, sample_size:int=10000):
        """
        Calculate p-value for estimated model parameters.

        :param z: z-statistics of the null hypotheses beta_i=0 for i = 0,1,2,...,k
        :type z: numpy.ndarray
        :param sample_size: Size of standard normal sample over which the p-values are computed given the z-statistics.
        :type sample_size: int, optional, defaults to 10000

        :return: p-values for the null hypotheses beta_i=0 for i = 0,1,2,...,k.
        :rtype: numpy.ndarray
        """
        normal_sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
        return np.array([np.where(normal_sample >= np.absolute(i), 1, 0).sum()/sample_size for i in z])

    def get_R2(self, y, res):
        """
        Compute and return R^2 - coefficient of determination.

        :param y: The fitted dependent variable (n x 1).
        :type y: numpy.ndarray
        :param res: The (n x 1) vector of the residual terms of a fitted model.
        :type res: numpy.ndarray

        :return: Coefficient of determination - R-squared.
        :rtype: float
        """
        return 1 - np.sum(res**2)/np.sum(y**2)
