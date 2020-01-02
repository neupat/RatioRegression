"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

# This Code is part of an implementation of a Sybolic Regression Algorithm
# directly using numerical data of the ratios of pairs of derivatives of the
# function one wants to find.
# For this purpose, Trevor Stephens original Code has been altered, because
# it was not possible without applying changes to the lowest levels of the code.

import numpy as np

__all__ = ['make_function']


class _Function:

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.


    diff_f : lambda function of either  arity 1: lambda a,c
                                    or  arity 2: lambda a,b,c,d
                                            where a and b are the subnodes,
                                            and c and d are the (inner) derivations of the nodes.
        The derivative of the nodes function, as a function of its subnodes and their derivatives.
    """

    def __init__(self, function, name, arity, diff_f=None):
        self.function = function
        self.name = name
        self.arity = arity
        if diff_f is not None:
            self.diff = diff_f
        else:
            self.differentiate()
    def __call__(self, *args):
        return self.function(*args)

    def differentiate(self):
        """
        todo:
        differentiate an unknown function and save derivative as diff
        """
        self.diff = None






def make_function(function, name, arity, diff_f=None):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x_1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    return _Function(function, name, arity, diff_f)


def _protected_division(x_1, x_2):
    """Closure of division (x_1/x_2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x_2) > 0.001, np.divide(x_1, x_2), 1.)


def _protected_sqrt(x_1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x_1))


def _protected_log(x_1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x_1) > 0.001, np.log(np.abs(x_1)), 0.)


def _protected_inverse(x_1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x_1) > 0.001, 1. / x_1, 0.)


def _sigmoid(x_1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x_1))


#todo diff_f for all functions and make diff_f non-default argument in make_function
ADD2 = make_function(function=np.add, name='add', arity=2,
                     diff_f=lambda a, b, c, d: c+d)
SUB2 = make_function(function=np.subtract, name='sub', arity=2,
                     diff_f=lambda a, b, c, d: c-d)
MUL2 = make_function(function=np.multiply, name='mul', arity=2,
                     diff_f=lambda a, b, c, d: a*d + b*c)
DIV2 = make_function(function=_protected_division, name='div', arity=2,
                     diff_f=lambda a, b, c, d: _protected_division(c*b - a*d, b**2))
SQRT1 = make_function(function=_protected_sqrt, name='sqrt', arity=1,
                      diff_f=lambda a, c: _protected_division(0.5, _protected_sqrt(a)*np.sign(a)*c))
LOG1 = make_function(function=_protected_log, name='log', arity=1)
NEG1 = make_function(function=np.negative, name='neg', arity=1)
INV1 = make_function(function=_protected_inverse, name='inv', arity=1)
ABS1 = make_function(function=np.abs, name='abs', arity=1)
MAX2 = make_function(function=np.maximum, name='max', arity=2)
MIN2 = make_function(function=np.minimum, name='min', arity=2)
SIN1 = make_function(function=np.sin, name='sin', arity=1,
                     diff_f=lambda a, c: np.cos(a)*c)
COS1 = make_function(function=np.cos, name='cos', arity=1,
                     diff_f=lambda a, c: -np.sin(a)*c)
TAN1 = make_function(function=np.tan, name='tan', arity=1)
SIG1 = make_function(function=_sigmoid, name='sig', arity=1)

ARCTAN1 = make_function(function=np.arctan, name='arctan', arity=1,
                        diff_f=lambda a, c: c/(1.+a**2))


_FUNCTION_MAP = {'add': ADD2,
                 'sub': SUB2,
                 'mul': MUL2,
                 'div': DIV2,
                 'sqrt': SQRT1,
                 'log': LOG1,
                 'abs': ABS1,
                 'neg': NEG1,
                 'inv': INV1,
                 'max': MAX2,
                 'min': MIN2,
                 'sin': SIN1,
                 'cos': COS1,
                 'tan': TAN1,
                 'arctan':ARCTAN1}
