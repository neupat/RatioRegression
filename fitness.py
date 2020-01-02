"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

# This Code is part of an implementation of a Sybolic Regression Algorithm
# directly using numerical data of the ratios of pairs of derivatives of the
# function one wants to find.
# For this purpose, Trevor Stephens original Code has been altered, because
# it was not possible without applying changes to the lowest levels of the code.

import numbers

import numpy as np
from scipy.stats import rankdata

__all__ = ['make_fitness']


class _Fitness:

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)

    def get_function(self):
        """
        get the function object itself, useful for debugging
        """
        return self.function


def make_fitness(function, greater_is_better):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                               np.array([2, 2]),
                               np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    return _Fitness(function, greater_is_better)


def _weighted_pearson(y_data, y_pred, weights):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=weights)
        y_demean = y_data- np.average(y_data, weights=weights)
        corr = ((np.sum(weights * y_pred_demean * y_demean) / np.sum(weights)) /
                np.sqrt((np.sum(weights * y_pred_demean ** 2) *
                         np.sum(weights * y_demean ** 2)) /
                        (np.sum(weights) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y_data, y_pred, weights):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y_data)
    return _weighted_pearson(y_pred_ranked, y_ranked, weights)


def _mean_absolute_error(y_data, y_pred, weights):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y_data), weights=weights)


def _mean_square_error(y_data, y_pred, weights):
    """Calculate the mean square error."""
    return np.average(((y_pred - y_data) ** 2), weights=weights)


def _root_mean_square_error(y_data, y_pred, weights):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y_data) ** 2), weights=weights))


def _log_loss(y_data, y_pred, weights):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y_data* np.log(y_pred) + (1 - y_data) * np.log(inv_y_pred)
    return np.average(-score, weights=weights)


WEIGHTED_PEARSON = make_fitness(function=_weighted_pearson,
                                greater_is_better=True)
WEIGHTED_SPEARMAN = make_fitness(function=_weighted_spearman,
                                 greater_is_better=True)
MEAN_ABSOLUTE_ERROR = make_fitness(function=_mean_absolute_error,
                                   greater_is_better=False)
MEAN_SQUARE_ERROR = make_fitness(function=_mean_square_error,
                                 greater_is_better=False)
ROOT_MEAN_SQUARE_ERROR = make_fitness(function=_root_mean_square_error,
                                      greater_is_better=False)
LOG_LOSS = make_fitness(function=_log_loss, greater_is_better=False)

_FITNESS_MAP = {'pearson': WEIGHTED_PEARSON,
                'spearman': WEIGHTED_SPEARMAN,
                'mean absolute error': MEAN_ABSOLUTE_ERROR,
                'mse': MEAN_SQUARE_ERROR,
                'rmse': ROOT_MEAN_SQUARE_ERROR,
                'log loss': LOG_LOSS}
