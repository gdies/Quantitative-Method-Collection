# IMPORT MODULES
import numpy as np
from scipy import optimize
from scipy import special
from reported_statistics import get_p
from typing import List
import time


class BinaryOutcomeModel(object):
    """
    A binary outcome model that Logit and Probit are built on
    """
    def __init__(self, add_intercept:bool=False, compute_statistics:bool=True, bootstrap_marginal_effect_variances:bool=False):
        """
        add_intercept       :   add column of ones to data
        compute_statistics  :   in case the model is not only for prediction
        bootstrap_marginal_effect_variances : default is using the delta method
        """
        self.parameters = {'add_intercept':add_intercept, 'compute_statistics':compute_statistics, 'bootstrap_marginal_effect_variances':bootstrap_marginal_effect_variances}
        self.X:np.ndarray = None
        self.y:np.ndarray = None
        self.dummy_columns:List[bool] = []
        self.beta:np.ndarray = None
        self.covariance_matrix = None

    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Fit binary model, using the BFGS algorithm for MLE (minimize the negative log-likelihood)
        """
        if self.parameters['add_intercept']:
            X = np.append(X, np.ones(shape=(X.shape[0],1), dtype=np.int8), axis=1)
        n, k = X.shape
        self.X, self.y = X, y
        self.get_dummy_columns() # dummy check

        # MLE - optimization
        beta_0 = np.zeros(shape=k)
        bfgs = optimize.minimize(self.objective_function, beta_0, method='BFGS', jac=self.score, options={'disp': True}) # MLE
        self.beta = bfgs.x

        if self.parameters['compute_statistics']:
            self.compute_statistics()

    def predict(self, X):
        """
        Predict probability: Pr(y=1|x)
        return y_hat and Pr(y=1|x)
        """
        p = self.link_function(X, self.beta)
        y_hat = np.where(p > 0.5, 1, 0)
        return y_hat, p

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
        self.p_values = get_p(self.beta, self.covariance_matrix)
        pass

    def get_covariance_matrix(self):
        """
        Compute covariance matrix for the parameters
        The asymptotical variance is the inverse information matrix
        """
        self.covariance_matrix = np.linalg.inv(self.information())
        return self.covariance_matrix

    def get_marginal_effects(self, X:np.ndarray, variances:bool=True):
        """
        Calculate marginal effects
        X (n x k) - must follow the structure of X
        Output: (n x k) - k marginal effects for the n observations
        variances:  - variances for the (n x k) marginal effects -> return (marginals, var_marginals)
        """
        marginals = self.compute_marginal_effects(X, self.beta) # (n x k)
        if variances:
            var_marginals = self.get_marginal_effect_variances(X) # (n x k)
            return marginals, var_marginals
        return marginals

    def compute_marginal_effects(self, X:np.ndarray, beta:np.ndarray):
        """
        Calculate marginal effects given X and beta
        """
        marginals = []
        for i in range(X.shape[0]): # for each row in X
            marginals_i = []
            for j in range(X.shape[1]): # for each column in X
                if self.dummy_columns[j]:  # difference of predicted probabilities (x_ij = 1 and x_ij = 0) (Dummy variables)
                    X_i1, X_i0 = X[i].copy(), X[i].copy()
                    X_i1[j], X_i0[j] = 1, 0
                    marginals_i.append(self.link_function(X_i1, beta) - self.link_function(X_i0, beta))
                else:  # derivative of predicted probability wrt. x_ij (Continuous variables)
                    marginals_i.append(self.link_function_derivative(X[i], beta)*beta[j]) # (1 x k)
            marginals.append(marginals_i)
        marginals = np.array(marginals) # (n x k)
        return marginals

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

    def bootstrap(self, X:np.ndarray):
        """
        Implement bootstrap method for marginal effect variance computation for Logit and Probit
        Bootstrap variance of marginal effects (X (n x k))
        Output: (n x k) - variances for k marginal effects for the n observations
        """
        start = time.time()
        # sample coefficients - from their marginal distribution, using the fact that the coefficients are asymptotically normal
        bootstrap_size = 500 # bootstrap size: 500
        beta_samples = np.zeros(shape=(bootstrap_size, X.shape[1])) # preallocate sample matrix (b x k)
        for j in range(X.shape[1]): # populate sample matrix
            beta_samples[:,j] = np.random.normal(loc=self.beta[j], scale=np.diag(self.covariance_matrix)[j], size=bootstrap_size) # add samples as columns

        # get marginal effects for each coefficient sample
        marginal_effect_samples = np.zeros(shape=(bootstrap_size, X.shape[0], X.shape[1])) # (b x n x k) tensor
        for b in range(bootstrap_size):
            marginal_effect_samples[b,:,:] = self.compute_marginal_effects(X, beta_samples[b,:])
        print(f'computing marginal effects for each coefficient sample took {time.time()-start} seconds')

        # variance of marginal effect samples (the mean is given - calculated in self.get_marginal_effects)
        variances = np.var(marginal_effect_samples, axis=0) # collapse to (n x k)
        return variances

# not implemented methods
    def objective_function(self, beta:np.ndarray):
        """
        negative log likelihood
        """
        link = self.link_function(self.X, beta)
        return - np.sum( self.y*np.log(link) + (1 - self.y)*np.log(1 - link) ) / self.X.shape[0]

    def link_function(self):
        """
        Implement link function for Logit or Probit
        """
        raise NotImplementedError('This method must be implemented for the specific binary outcome model')

    def link_function_derivative(self):
        """
        Implement the derivative link function for Logit or Probit
        """
        raise NotImplementedError('This method must be implemented for the specific binary outcome model')

    def delta_method(self):
        """
        Implement delta method for marginal effect variance computation for Logit or Probit
        """
        raise NotImplementedError('This method must be implemented for the specific binary outcome model')

    def score(self):
        """
        Implement objective function derivative
        """
        raise NotImplementedError('This method must be implemented for the specific binary outcome model')

    def information(self):
        """
        Implement Fisher Information matrix computation
        """
        raise NotImplementedError('This method must be implemented for the specific binary outcome model')


class Logit(BinaryOutcomeModel):
    """
    Fit logit
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def link_function(self, X:np.ndarray, beta:np.ndarray):
        """
        probit or logit, can be used for prediction, so does not use self.X as default
        """
        return (np.exp(X @ beta)) / (1 + np.exp(X @ beta)) # logit

    def link_function_derivative(self, X:np.ndarray, beta:np.ndarray):
        """
        The derivative of the link function
        """
        link = self.link_function(X, beta)
        return link * (1 - link)

    def score(self, beta:np.ndarray):
        """
        Return the first derivative of the objective function at beta
        """
        res = self.y - self.link_function(self.X, beta)
        return - np.sum( self.X * np.reshape(res, newshape=(res.shape[0], 1)), axis=0 )

    def information(self):
        """
        Compute Fisher information matrix
        """
        information = np.zeros(shape=(self.X.shape[1], 1))
        for i in range(self.X.shape[0]):
            X_i = np.reshape(self.X[i], newshape=(1, self.X[i].shape[0])) # X_i is a row vector
            link_i = self.link_function(X_i, self.beta)
            w = (link_i) * (1 - link_i) # or just link derivative
            information = information + w * (np.transpose(X_i) @ X_i)
        return information

    def delta_method(self, X:np.ndarray):
        """
        Calculate variance of marginal effects (X (n x k))
        Output: (n x k) - variances for k marginal effects for the n observations
        """
        # continuous variables
        link = self.link_function(X, self.beta) # (n x 1)
        link_derivative = self.link_function_derivative(X, self.beta) # (n x 1)
        variances_list = []
        for i in range(X.shape[0]): # go through each row of X
            X_i = np.reshape(X[i], newshape=(1, X[i].shape[0])) # X_i is a row vector
            partial = np.identity(X.shape[1]) * link_derivative[i] + np.outer((self.beta * link_derivative[i] * (1 - 2*(link[i])) ), X_i) # (k x k)
            variances_i = list(np.diag(partial @ self.covariance_matrix @ np.transpose(partial))) # 1 x k

            # binary variables
            for j in range(len(variances_i)):
                if self.dummy_columns[j]: # replace variance if dummy
                    X_i1, X_i0 = X_i.copy(), X_i.copy()
                    X_i1[0,j], X_i0[0,j] = 1, 0
                    partial = link_derivative[i] * X_i1 - link_derivative[i] * X_i0
                    variances_i[j] = (partial) @ self.covariance_matrix @ np.transpose(partial)

            variances_list.append(variances_i) # construct list of variance lists
        return np.array(variances_list, dtype=object) # the variances (n x k)


class Probit(BinaryOutcomeModel):
    """
    Fit probit
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def link_function(self, X:np.ndarray, beta:np.ndarray):
        """
        probit -> standard normal distribution, can be used for prediction, so does not use self.X as default
        """
        return ((1 + special.erf((X @ beta) / np.sqrt(2))) / 2) # Probit

    def link_function_derivative(self, X:np.ndarray, beta:np.ndarray):
        """
        The derivative of the link function
        """
        return np.exp(- ((X @ beta)**2) / 2) / np.sqrt(2*np.pi)

    def score(self, beta:np.ndarray):
        """
        Return the first derivative of the objective function at beta
        """
        link, link_derivative = self.link_function(self.X, beta), self.link_function_derivative(self.X, beta)
        res = self.y - link
        w = link_derivative / ((link)*(1 - link))
        return - np.sum(self.X * np.reshape(res, newshape=(res.shape[0], 1)) * np.reshape(w, newshape=(res.shape[0], 1)), axis=0 )

    def information(self):
        """
        Compute Fisher information matrix
        """
        information = np.zeros(shape=(self.X.shape[1], 1))
        for i in range(self.X.shape[0]):
            X_i = np.reshape(self.X[i], newshape=(1, self.X[i].shape[0])) # X_i is a row vector
            link_i, link_derivative_i = self.link_function(X_i, self.beta), self.link_function_derivative(X_i, self.beta)
            w = link_derivative_i**2 / ((link_i)*(1 - link_i))
            information = information + w * (np.transpose(X_i) @ X_i)
        return information

    def delta_method(self, X:np.ndarray):
        """
        Calculate variance of marginal effects (X (n x k))
        Output: (n x k) - variances for k marginal effects for the n observations
        """
        # continuous variables
        link_derivative = self.link_function_derivative(X, self.beta) # (n x 1)
        variances_list = []
        for i in range(X.shape[0]): # go through each row of X
            X_i = np.reshape(X[i], newshape=(1, X[i].shape[0])) # X_i is a row vector
            partial = link_derivative[i] * (np.identity(X.shape[1]) - np.outer(np.outer(self.beta, np.transpose(self.beta)) @ np.transpose(X_i), X_i)) # (k x k)
            variances_i = list(np.diag(partial @ self.covariance_matrix @ np.transpose(partial))) # 1 x k

            # binary variables
            for j in range(len(variances_i)):
                if self.dummy_columns[j]: # replace variance if dummy
                    X_i1, X_i0 = X_i.copy(), X_i.copy()
                    X_i1[0,j], X_i0[0,j] = 1, 0
                    partial = link_derivative[i] * X_i1 - link_derivative[i] * X_i0
                    variances_i[j] = (partial) @ self.covariance_matrix @ np.transpose(partial)

            variances_list.append(variances_i) # construct list of variance lists
        return np.array(variances_list, dtype=object) # the variances (n x k)
