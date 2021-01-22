# IMPORT MODULES
import numpy as np
from scipy import optimize
from reported_statistics import get_p
from typing import List
import time
import distributions



class Tobit(object):
    """
    A regression model for censored or truncated variables
    """
    def __init__(self, add_intercept:bool=False, compute_statistics:bool=True):
        """
        add_intercept       :   add column of ones to data
        compute_statistics  :   in case the model is not only for prediction
        """
        self.parameters = {'add_intercept':add_intercept, 'compute_statistics':compute_statistics}
        self.X:np.ndarray = None
        self.y:np.ndarray = None
        self.dummy_columns:List[bool] = []
        self.theta:np.ndarray = None # {beta, sigma}
        self.beta:np.ndarray = None
        self.sigma:np.ndarray = None
        self.p_values:np.ndarray = None
        self.covariance_matrix = None

    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Fit censored model, using the BFGS algorithm for MLE (minimize the negative log-likelihood)
        """
        if self.parameters['add_intercept']:
            X = np.append(X, np.ones(shape=(X.shape[0],1), dtype=np.int8), axis=1)
        n, k = X.shape
        self.X, self.y = X, y
        self.get_dummy_columns() # dummy check

        # MLE - optimization
        theta_0 = np.random.normal(loc=0, scale=1, size=(k+1)) # starting values for the parameter vector
        theta_0[k] = np.abs(theta_0[k]) # sigma cannot be negative
        bfgs = optimize.minimize(self.objective_function, theta_0, method='BFGS', jac=self.score, options={'disp': True}) # MLE
        self.theta = bfgs.x
        self.beta, self.sigma = self.theta[0:len(self.theta)-1], self.theta[-1]

        if self.parameters['compute_statistics']:
            self.compute_statistics()

    def predict(self, X, out:str='latent'):
        """
        Predict out:
            'latent'        : E[y* | X]
            'censored'      : E[y | X]
            'truncated'     : E[y | y>0,X]
            'P(censored)'   : P(y=0 | X)
            'P(uncensored)' : P(y>0 | X)
        """
        beta, sigma = self.theta[0:len(self.theta)-1], self.theta[-1]
        normal = distributions.Normal(mu=0, sigma=1)
        XB = self.X @ beta
        cdf_eval, pdf_eval = normal.cdf(XB / sigma), normal.pdf(XB / sigma)
        if out == 'latent':
            return XB
        elif out == 'censored':
            return XB * cdf_eval + sigma * pdf_eval
        elif out == 'truncated':
            return XB + sigma * (pdf_eval/cdf_eval)
        elif out == 'P(censored)':
            return 1 - cdf_eval
        elif out == 'P(uncensored)':
            return cdf_eval
        else:
            raise ValueError('"out" argument must be one of ["latent", "censored", "truncated", "P(censored)", "P(uncensored)"]')

    def get_dummy_columns(self):
        """
        refresh dummy_columns list - tells which column in X is dummy variable, which one is continuous
        """
        self.dummy_columns = []
        for i in range(self.X.shape[1]):
            if sorted(list(np.unique(self.X[:,i]))) == [0, 1]: # if dummy
                self.dummy_columns.append(True)
            else:
                self.dummy_columns.append(False)

    def compute_statistics(self):
        """
        Compute standard error
        """
        self.covariance_matrix = self.get_covariance_matrix()
        self.p_values = get_p(self.theta, self.covariance_matrix)
        pass

    def get_covariance_matrix(self, cov_type:str='sandwich'):
        """
        Compute covariance matrix for the parameters
        The asymptotical variance is the inverse information matrix
        """
        N = self.X.shape[0]
        if cov_type == 'sandwich':
            A_inv = np.linalg.inv( self.hessian(self.theta) )
            B = self.outer_product(self.theta)
            self.covariance_matrix = 1/N * A_inv @ B @ A_inv
        elif cov_type == 'outer':
            B_inv = np.linalg.inv( self.outer_product(self.theta) )
            self.covariance_matrix = 1/N * B_inv
        elif cov_type == 'hessian':
            A_inv = np.linalg.inv( self.hessian(self.theta) )
            self.covariance_matrix = 1/N * A_inv
        else:
            raise ValueError('cov_type argument only accepts "sandwich", "outer" or "hessian" as value.')
        return self.covariance_matrix

    # unfinished
    def get_marginal_effects(self, X:np.ndarray, variances:bool=True):
        """
        Calculate marginal effects
        X (n x k) - must follow the structure of X
        Output: (n x k) - k marginal effects for the n observations
            variances:  - variances for the (n x k) marginal effects -> return (marginals, var_marginals)
        """
        marginals = self.compute_marginal_effects(X, self.theta) # (n x k)
        if variances:
            var_marginals = self.get_marginal_effect_variances(X) # (n x k)
            return marginals, var_marginals
        return marginals

    def compute_marginal_effects(self, X:np.ndarray, theta:np.ndarray, out:str='latent'):
        """
        Calculate marginal effects given X and theta
        out:
            'latent'    : dE[y*|X]/dx
            'censored'  : dE[y|X]/dx
        """
        if out == 'latent':
            return theta[0:len(theta)-1] # beta
        elif out == 'censored':
            beta, sigma = theta[0:len(theta)-1], theta[-1]
            normal = distributions.Normal(mu = 0, sigma = 1)
            return normal.cdf((X @ beta)/sigma)*beta
        else:
            raise ValueError('Input argument "out" must be either "latent" or "censored"')

    # unfinished
    def get_marginal_effect_variances(self, X:np.ndarray):
        """
        Calculate variance of marginal effects
        X (n x k)
        Output: (n x k) - variances for k marginal effects for the n observations
        """
        if self.parameters['bootstrap_marginal_effect_variances']: # bootsrap method
            variances = self.bootstrap(X) # n x k
        else: # delta method
            variances = self.delta_method(X) # n x k
        return variances

    # unfinished
    def bootstrap(self, X:np.ndarray):
        """
        Implement bootstrap method for marginal effect variance computation for Logit and Probit
        Bootstrap variance of marginal effects (X (n x k))
        Output: (n x k) - variances for k marginal effects for the n observations
        """
        pass

# not implemented methods
    def objective_function(self, theta:np.ndarray):
        """
        negative log likelihood
        theta: {beta, sigma}
        """
        beta, sigma = theta[0:len(theta)-1], theta[-1]
        normal = distributions.Normal(mu = 0, sigma = 1)
        XB = self.X @ beta
        f_0 = (self.y==0) * np.log(1 - normal.cdf(XB / sigma))
        f_1 = (self.y>0) * ((-1/2) * np.log(2 * np.pi) - np.log(sigma**2) / 2 - (self.y - XB)**2 / (2 * sigma**2))
        return (-1) * np.sum(f_0 + f_1, axis=0) / self.X.shape[0] # sum log likelihood (meaned)

    # may not be necessary
    def delta_method(self):
        """
        Implement delta method for marginal effect variance computation for Logit or Probit
        """
        raise NotImplementedError('This method must be implemented for the specific binary outcome model')

    def score(self, theta:np.ndarray):
        """
        Implement objective function derivative
        """
        beta, sigma = theta[0:len(theta)-1], theta[-1]
        normal = distributions.Normal(mu = 0, sigma = 1)
        XB = self.X @ beta
        r = self.y - XB
        cdf_eval, pdf_eval = normal.cdf(XB / sigma), normal.pdf(XB / sigma)
        partial_beta = (1/sigma**2) * np.sum( np.reshape(((self.y > 0) * r - (self.y == 0) * ((sigma * pdf_eval) / (1 - cdf_eval))), newshape=(-1,1)) * self.X , axis=0) # (K x 1)
        partial_sigma = np.sum( (self.y > 0) * ((-1) / (2*sigma**2) + r**2 / (2*sigma**4)) + (self.y == 0) * XB / (2*sigma**3) * pdf_eval / (1 - cdf_eval) ) # (1 x 1)
        #print('score: ', np.concatenate((partial_beta, np.atleast_1d(partial_sigma)), axis=0))
        return - np.concatenate((partial_beta, np.atleast_1d(partial_sigma)), axis=0) # the full score vector ((K+1) x 1)

    def hessian(self, theta:np.ndarray):
        """
        Compute and construct hessian at theta,
        the formulae found in Calzolari & Fiorentini 1993
        """
        beta, sigma = theta[0:len(theta)-1], theta[-1]
        normal = distributions.Normal(mu = 0, sigma = 1)
        XB = self.X @ beta
        r = self.y - XB
        cdf_eval, pdf_eval = normal.cdf(XB / sigma), normal.pdf(XB / sigma)

        # Hessian blocks
        dL_dBdB = np.zeros(shape=(self.X.shape[1], self.X.shape[1])) # (K x K), d^2L/dBdB'
        dL_dsdB = np.zeros(shape=(self.X.shape[1])) # (K x 1), d^2L/ds^2dB'
        dL_dsds = np.zeros(shape=(1)) # (1 x 1), d^2L/d^2s^2
        for i in range(self.X.shape[0]):
            xixi = np.outer(self.X[i,:], self.X[i,:]) # (K x K)
            dL_dBdB += (-1) * (self.y[i]>0) * (1/sigma**2) * xixi - (self.y[i]==0) * (1/sigma) * (pdf_eval[i]/(1-cdf_eval[i])**2) * ((pdf_eval[i]/sigma) - (1/sigma**2)*(1-cdf_eval[i])*XB[i]) * xixi # (K x K)
            dL_dsdB += (-1) * (self.y[i]>0) * (1/sigma**2) * r[i] * self.X[i,:] - (self.y[i]==0) * (1/(2*sigma**3)) * (pdf_eval[i]/(1 - cdf_eval[i])**2) * ((1/sigma**2)*(1-cdf_eval[i])*XB[i]**2 - (1-cdf_eval[i]) - ((XB[i]*pdf_eval[i])/sigma)) * self.X[i,:] # (K x 1)
            dL_dsds += (-1) * (self.y[i]>0) * (1/sigma**6) * r[i]**2 - (self.y[i]==0) * (1/(4*sigma**5)) * (pdf_eval[i]/(1-cdf_eval[i])**2) * ((1/sigma**2)*(1-cdf_eval[i])*(XB[i]**3) - 3*(1-cdf_eval[i])*(XB[i]) - (XB[i]*pdf_eval[i]/sigma)) # (1 x 1)
        return np.concatenate(( np.concatenate((dL_dBdB, np.reshape(dL_dsdB, newshape=(-1, 1))), axis=1), np.reshape(np.concatenate((dL_dsdB, dL_dsds), axis=0), newshape=(1, -1)) ), axis=0)

    def outer_product(self, theta:np.ndarray):
        """
        Compute outer product of gradients at theta
        the formulae found in Calzolari & Fiorentini 1993
        """
        beta, sigma = theta[0:len(theta)-1], theta[-1]
        normal = distributions.Normal(mu = 0, sigma = 1)
        XB = self.X @ beta
        r = self.y - XB
        cdf_eval, pdf_eval = normal.cdf(XB / sigma), normal.pdf(XB / sigma)

        # Outer Product blocks
        dL_dBdB = np.zeros(shape=(self.X.shape[1], self.X.shape[1])) # (K x K), d^2L/dBdB'
        dL_dsdB = np.zeros(shape=(self.X.shape[1])) # (K x 1), d^2L/ds^2dB'
        dL_dsds = np.zeros(shape=(1)) # (1 x 1), d^2L/d^2s
        for i in range(self.X.shape[0]):
            xixi = np.outer(self.X[i,:], self.X[i,:]) # (K x K)
            dL_dBdB += (-1) * (1/sigma**2) * (pdf_eval[i]*(XB[i]/sigma) - (pdf_eval[i]**2 / (1 - cdf_eval[i])) - cdf_eval[i]) * xixi
            dL_dsdB += (-1) * (1/(2*sigma**3)) * (pdf_eval[i] * (XB[i]/sigma)**2 + pdf_eval[i] - (pdf_eval[i]**2 / (1-cdf_eval[i])) * (XB[i]/sigma)) * self.X[i,:]
            dL_dsds += (-1) * (1/(4*sigma**4)) * (pdf_eval[i] * (XB[i]/sigma)**3 + pdf_eval[i] * (XB[i]/sigma) - (pdf_eval[i]**2/(1-cdf_eval[i])) * (XB[i]/sigma) - 2*cdf_eval[i])
        return np.concatenate(( np.concatenate((dL_dBdB, np.reshape(dL_dsdB, newshape=(-1, 1))), axis=1), np.reshape(np.concatenate((dL_dsdB, dL_dsds), axis=0), newshape=(1, -1)) ), axis=0)

    # not necessary
    def information(self):
        """
        Implement Fisher Information matrix computation
        """
        raise NotImplementedError('This method must be implemented for the specific model')
