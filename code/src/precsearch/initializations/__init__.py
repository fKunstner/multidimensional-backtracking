import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class Initializer:
    """
    An abstract base class for initialization methods.

    Methods:
        initialize(X, y): A method that takes in a feature matrix `X` and a target vector
            `y` and returns an initial weight vector for a machine learning model.
    """

    def initialize(self, X, y):
        """
        Initializes the weight vector for a machine learning model.

        Args:
            X: A feature matrix of shape (n_samples, n_features).
            y: A target vector of shape (n_samples,).

        Returns:
            A weight vector of shape (n_features,).
        """
        raise NotImplementedError

    def uname(self):
        return self.__class__.__name__


@dataclass
class GaussianInitializer(Initializer):
    var: float = 1
    seed: int = 0

    def initialize(self, X, y):
        """
        Initializes the weight vector for a machine learning model from a Gaussian distribution.

        Args:
            X: A feature matrix of shape (n_samples, n_features).
            y: A target vector of shape (n_samples,).

        Returns:
            A weight vector of shape (n_features,).
        """
        warnings.warn("Random seed reinitialized to {self.seed}")
        np.random.seed(self.seed)
        dim = X.shape[1]
        w = np.random.randn(dim) * np.sqrt(self.var)
        return w


@dataclass
class BiasInitializerLogReg(Initializer):
    def initialize(self, X, y):
        """
        Initializes the bias vector for a logistic regression by setting
        the weights to 0 except for the bias term, where it sets it to the MLE
        (conditioned on the other weights being 0).

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data matrix.
        y : np.ndarray, shape (n_samples,)
            The target vector containing 0 or 1.

        Returns:
        --------
        w : np.ndarray, shape (n_features,)
            The initialized bias vector.

        Notes:
        ------
        This implementation sets the bias vector to the maximum likelihood estimate for a logistic regression with no
        features, which is given by the log-odds of the average target value clipped to the range [0.01, 0.99].
        """
        dim = X.shape[1]
        w = np.zeros((dim))
        clipped_avg = np.clip(np.mean(y), 0.01, 0.99)
        w[dim - 1] = np.log(clipped_avg / (1 - clipped_avg))
        return w


@dataclass
class BiasInitializerLinReg(Initializer):
    def initialize(self, X, y):
        """
        Initializes the bias vector for a logistic regression by setting
        the weights to 0 except for the bias term, where it sets it to the MLE
        (conditioned on the other weights being 0).

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data matrix.
        y : np.ndarray, shape (n_samples,)
            The target vector containing 0 or 1.

        Returns:
        --------
        w : np.ndarray, shape (n_features,)
            The initialized bias vector.

        Notes:
        ------
        This implementation sets the bias vector to the maximum likelihood estimate for a logistic regression with no
        features, which is given by the log-odds of the average target value clipped to the range [0.01, 0.99].
        """
        dim = X.shape[1]
        w = np.zeros((dim))
        w[dim - 1] = np.mean(y)
        return w
