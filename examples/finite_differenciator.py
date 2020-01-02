"""
Taking finite difference derivative ratios of a function versus all its arguments.
Can be improved significantely by using more elaborate finite difference formulas.
Only for the time being.
"""

import numpy as np

from ratio_regression.program import save_div


def get_ratio(f, X, nom, denom, method='finite_difference', **kwargs):
    """
    :param f: python function taking an array of length n_features = X.shape(1) as first argument and maybe **kwargs as second
    :param X: Evaluation points, shape = (n_data_points, n_features)
    :param nom: The nominator derivative will be taken w.r.t to the variable with index "nom", i.e. X[:,nom]
    :param denom: The denominator derivative will be taken w.r.t to the variable with index "denom", i.e. X[:,denom]
    :param kwargs: Extra arguments to the function f, cannot have keyword "method"!!
    :return: Finite difference derivative ratio R at points X, R.shape = (n_data_points,)
    """
    df_dnom = get_derivative(f, X.astype(float), nom, method, kwargs)
    df_ddenom = get_derivative(f,X.astype(float),denom, method, kwargs)

    return save_div(df_dnom,df_ddenom)


def get_derivative(f, X, index, method, kwargs):
    """
    :param f: python function taking an array of length n_features = X.shape(1) as first argument and maybe **kwargs as second
    :param X: Evaluation points, shape = (n_data_points, n_features)
    :param index: The derivative will be taken w.r.t to the variable with index "index", i.e. X[:,index]
    :param kwargs: Extra arguments to the function f, cannot have keyword "method"!!
    :return: Finite difference derivative at points X w.r.t variable "index", df.shape = (n_data_points,)
    """
    eval_point_diffs = X[1:, index] - X[:-1, index]
    avg_diff = np.mean(eval_point_diffs)
    dt = 1e-5*avg_diff
    if method == 'finite_difference':
        X[:, index] += dt
        top = f(X, **kwargs)
        X[:, index] -= 2*dt
        bottom = f(X, **kwargs)
        return (top - bottom)/(2*dt)
    raise NotImplemented


if __name__ == '__main__':
    def f(X,**kwargs):
        print(kwargs.get('message'))
        val = np.sin(X[:,0])*X[:,1]
        return val/X[:,2]

    X = (np.arange(3*50)/2.3).reshape(-1,3)
    print(get_ratio(f, X, 0, 2, message='hello'))
