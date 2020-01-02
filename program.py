"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

# This Code is part of an implementation of a Sybolic Regression Algorithm
# directly using numerical data of the ratios of pairs of derivatives of the
# function one wants to find.
# For this purpose, Trevor Stephens original Code has been altered, because
# it was not possible without applying changes to the lowest levels of the code.

from copy import copy
import random

import numpy as np
from sklearn.utils.random import sample_without_replacement
from lmfit import Minimizer, Parameters

from ratio_regression.functions import  _Function
from ratio_regression.utils import check_random_state, INFINITY_VALUE


def save_div(x_array, y_array, thresh=1e-9):
    """
    customised save division. Sometimes it is important to choose the threshold
    w.r.t to the o.o.m of the expected noise on the data
    the value of \\infty and \frac{1}{\\infty} is important, as in
    some cases it will be checked for equality.
    Therefore it is globally specified in genetic.py
    """
    if not isinstance(y_array, np.ndarray):
        y_array = np.array(y_array)
    if not isinstance(x_array, np.ndarray):
        y_array = np.array(x_array)
    if np.array_equal(x_array, y_array):
        if np.signbit(x_array).any() == np.signbit(y_array).any():
            return np.ones_like(x_array)
        return -np.ones_like(x_array)
    _ratio = x_array/y_array
    _ratio[np.abs(y_array) < thresh] = np.sign(y_array[np.abs(y_array) < thresh])*\
                                       np.sign(x_array[np.abs(y_array) < thresh])*INFINITY_VALUE
    _ratio[np.abs(y_array) == 0] = INFINITY_VALUE
    return _ratio



class _Program:

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 int_pars=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.int_pars = int_pars

        if self.program is not None:
            if not self.validate_program():
                print(self.program)
                print(self)
                raise ValueError('The supplied program is incomplete.')
            if self.int_pars is None:
                self.int_pars = [1. for x in self.program if isinstance(x, int)]

        else:
            # Create a naive random program
            self.program = self.build_program(random_state)
            self.int_pars = [1. for x in self.program if isinstance(x, int)]

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        program = [function]
        terminal_stack = [function.arity]


        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')

                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def optimize_n_vars(self, X, y, n=2, subsamp=0.2):
        """
        Optimizing n variable factors in a program on a subsample of the given data
        :param X: X-data, shaped (n_data_points, n_features)
        :param y: ratio-data to fit variable factors to, shaped (n_data_points,)
        :param n: how many variable factors should be optimized at once
        :param subsamp: on what ratio of the data point the fitting should be evaluated
        :return: None, self.int_pars is updated
        """
        p_0 = self.int_pars
        subsamp = (np.random.rand(len(y)) + float(subsamp)).astype(int)
        X = X[subsamp == 1, :]
        y = y[subsamp == 1]


        if n < len(p_0):
            indices = random.sample(list(np.arange(len(p_0))), n)
        elif len(p_0) >= 1:
            indices = list(np.arange(len(p_0)))
        else:
            return None
        lm_paras = Parameters()
        for i, para in enumerate(p_0):
            lm_paras.add(name="int_par" + str(i), value=para, vary=(i in indices), min=-10, max=10)
        def f_local(params, X, y):
            p_bar = np.array(list(params.values()))
            return np.abs(self.ratio_evaluate(X, int_pars=p_bar) - y)

        minner = Minimizer(f_local, lm_paras, fcn_args=(X, y))
        result = minner.minimize(maxfev=20)
        self.int_pars = [x.value for x in list(result.params.values())]

        return None

    def optimize_n_consts(self, X, y, n=2, subsamp=0.2, max_evals=20):
        """
        Optimizing n constants in a program to a subsample of the given data
        :param X: X-data, shaped (n_data_points, n_features)
        :param y: ratio-data to fit constants to, shaped (n_data_points,)
        :param n: how many constants should be optimized at once
        :param subsamp: on what ratio of the data point the fitting should be evaluated
        :param max_evals: maximum number of evaluations for the lmfit optimizer
        :return: None, self.int_pars is updated
        """
        subsamp = (np.random.rand(len(y)) + float(subsamp)).astype(int)
        X = X[subsamp == 1, :]
        y = y[subsamp == 1]


        p_0 = np.array([x for x in self.program if isinstance(x, float)])
        if n < len(p_0):
            indices = random.sample(list(np.arange(len(p_0))), n)
        elif p_0.size:
            indices = list(np.arange(len(p_0)))
        else:
            return None
        lm_paras = Parameters()
        for i, para in enumerate(p_0):
            lm_paras.add(name="par" + str(i), value=para, vary=(i in indices), min=-10., max=10.)
        def f_local(params, X, y):
            p_bar = np.array(list(params.values()))
            return np.abs(self.ratio_evaluate(X, paras=p_bar) - y)

        minner = Minimizer(f_local, lm_paras, fcn_args=(X, y))
        result = minner.minimize(maxfev=max_evals)
        p_iter = iter([x.value for x in list(result.params.values())])
        self.program = [item if not isinstance(item, float)
                        else p_iter.__next__() for item in self.program]

        return None




    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        k = 0
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += '%.2f'%float(self.int_pars[k]) +'*'+ 'X%s' % node
                    else:
                        output += '%.2f'%float(self.int_pars[k]) +'*'+self.feature_names[node]
                    k += 1
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    #todo graphviz picture with coefficients before variables
    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def cleanup(self, prob=1):
        """
        Cleaning up a program, deleting unnecessary code snippets with a certain probability
        :param prob: Probability with which code snippets are deleted
        """
        i = len(self.program)-1
        para_indices = [iter(self.int_pars).__next__() if isinstance(x, int)
                        else "n" for x in self.program]
        prints = False
        while i >= 0:
            if np.random.rand(1) < prob:
                node = self.program[i]
                if isinstance(node, _Function):
                    apply = []
                    low, high = self.get_specific_subtree(i+1)
                    apply.append(self.program[low])
                    if node.arity == 1:
                        if isinstance(apply[0], float):
                            self.program[i] = float(node(apply[0]))
                            del self.program[i + 1]
                        i -= 1
                        continue

                    apply.append(self.program[high])

                    if all(isinstance(n, float) for n in apply):
                        self.program[i] = float(node(*apply))
                        del self.program[i+1:i+1+node.arity]

                    elif all(isinstance(n, int) for n in apply) and apply[0] == apply[1]:
                        if node.name == "sub":
                            # prints=True
                            self.program[i] = float(0.)
                            del self.program[i + 1:i + 3]
                            para_indices[i+1] = "n"
                            para_indices[i+2] = "n"
                        elif node.name == "div":
                            # prints=True
                            self.program[i] = float(1.)
                            del self.program[i + 1:i + 3]
                            para_indices[i+1] = "n"
                            para_indices[i+2] = "n"
                        else:
                            i -= 1

                    elif (isinstance(apply[0], float) and apply[0] == 0.):
                        if node.name == "mul" or node.name == "div":
                            other_low, other_high = self.get_specific_subtree(i)
                            self.program[i] = float(0.)
                            del self.program[other_low + 1:other_high]
                        elif node.name == "add":
                            del self.program[low]
                            del self.program[i]
                        elif node.name == "sub" and isinstance(self.program[high], float):
                            self.program[high] = -self.program[high]
                            del self.program[low]
                            del self.program[i]

                        else:
                            i -= 1


                    elif (isinstance(apply[1], float) and apply[1] == 0.):
                        if node.name == "mul":
                            other_low, other_high = self.get_specific_subtree(i)
                            self.program[i] = float(0.)
                            del self.program[other_low + 1:other_high]
                        elif node.name == "add" or node.name == "sub":
                            del self.program[high]
                            del self.program[i]

                        elif node.name == "div":
                            raise ValueError("division bei Null")
                        else:
                            i -= 1

                    else:
                        i -= 1

                else:
                    i -= 1
            else:
                i -= 1
        if prints:
            print(para_indices)
            print(self.int_pars)
        self.int_pars = [el for el in para_indices if str(el) != "n"]
        if prints:
            print(self.int_pars)
            print(self.program)

    def ratio_evaluate(self, X_i, nom=0, denom=1, paras=None, int_pars=None):   #p=None as argument
        """Evaluate the ratio \frac{\\partial f / \\partial X_1}{\\partial f / \\partial X_2}
        for the program (f) according to X.

        Parameters
        ----------
        X_i :        {array-like}, shape = [n_samples, n_features]
                    Training vectors, where n_samples is the number of samples and
                    n_features is the number of features.

        nom:        {integer} specifying w.r.t which variable the nominator has to be derived

        denom:      {integer} specifying w.r.t which variable the denominator has to be derived

        paras:      {list} With which constant parameters (terminals) the program should evaluated.
                    Default None.
                    If None, the programs own parameters are used.
                    This is only needed to easily construct a wrapper function
                    for the least square optimization.

        int_pars:   {list} With which variable parameters the program should evaluated.
                    Default None.
                    If None, the programs own parameters are used.
                    This is only needed to easily construct a wrapper function
                    for the least square optimization.


        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of evaluating the program ratio on X.

        """

        #Checking for one node programs. If float,
        #0/0 is set to 0. (Maybe think about that, but is not too important)
        #If integer, i.e. corresponding to a variable,
        #0/0 := 1 and 1/0 := infinity_value
        X = X_i.copy()
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(0, X.shape[0])
        if isinstance(node, int):
            return np.repeat(float(node == nom)*INFINITY_VALUE, X.shape[0])

        apply_stack = []
        nom_stack = []
        denom_stack = []

        #todo more elegant solution than 3 iterators
        if int_pars is None:
            int_pars = self.int_pars
        int_iter1 = iter(int_pars[::-1])
        int_iter2 = iter(int_pars[::-1])
        int_iter3 = iter(int_pars[::-1])


        k = 0

        for node in self.program:
            if isinstance(node, _Function):
                apply_stack.append([node])
                # para_stack.append([p[j]])
                nom_stack.append([node.diff])
                denom_stack.append([node.diff])
            elif isinstance(node, float) and paras is not None:
                apply_stack[-1].append(paras[k])
                # para_stack[-1].append(p[j])
                nom_stack[-1].append(paras[k])
                denom_stack[-1].append(paras[k])
                k += 1


            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)
                # para_stack[-1].append(p[j])
                nom_stack[-1].append(node)
                denom_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                nom_fun = nom_stack[-1][0]
                denom_fun = denom_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t]*next(int_iter1) if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]

                nom_terminals = [np.repeat(float(0), X.shape[0]) if isinstance(t, float)
                                 else np.repeat(float(t == nom), X.shape[0])*next(int_iter2)
                                 if isinstance(t, int)
                                 else t for t in nom_stack[-1][1:]]

                denom_terminals = [np.repeat(float(0), X.shape[0]) if isinstance(t, float)
                                   else np.repeat(float(t == denom), X.shape[0])*next(int_iter3)
                                   if isinstance(t, int)
                                   else t for t in denom_stack[-1][1:]]

                intermediate_result = function(*terminals)
                nom_int_result = nom_fun(*(terminals + nom_terminals))
                denom_int_result = denom_fun(*(terminals + denom_terminals))

                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                    nom_stack.pop()
                    nom_stack[-1].append(nom_int_result)
                    denom_stack.pop()
                    denom_stack[-1].append(denom_int_result)
                else:
                    return save_div(nom_int_result, denom_int_result)

        # We should never get here
        return None


    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight, nom=0, denom=1, paras=None, int_pars=None):
        """Evaluate the raw fitness of the program according to X, y and maybe specified parameters.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        nom:        {integer} specifying w.r.t which variable the nominator has to be derived

        denom:      {integer} specifying w.r.t which variable the denominator has to be derived

        paras:      {list} With which constant parameters (terminals) the program should evaluated.
                    Default None.
                    If None, the programs own parameters are used.
                    This is only needed to easily construct a wrapper function
                    for the least square optimization.

        int_pars:   {list} With which variable parameters the program should evaluated.
                    Default None.
                    If None, the programs own parameters are used.
                    This is only needed to easily construct a wrapper function
                    for the least square optimization.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.ratio_evaluate(X, nom=nom, denom=denom, paras=paras, int_pars=int_pars)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def get_specific_subtree(self, start):
        """get the subtree that starts at node/ index "start" """
        stack = 1
        end = start
        while stack > end - start:
            node = self.program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program), copy(self.int_pars)

    def crossover(self, donor, random_state, donor_pars):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)

        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor

        n_1 = sum(1 for x in self.program[:start] if isinstance(x, int))
        n_2 = sum(1 for x in self.program[:end] if isinstance(x, int))
        m_1 = sum(1 for x in donor[:donor_start] if isinstance(x, int))
        m_2 = sum(1 for x in donor[:donor_end] if isinstance(x, int))
        new_pars = self.int_pars[:n_1] + donor_pars[m_1:m_2] + self.int_pars[n_2:]

        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), new_pars, removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        chicken_pars = [1. for x in chicken if isinstance(x, int)]
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state, chicken_pars)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))



        n_1 = sum(1 for x in self.program[:start] if isinstance(x, int))
        n_2 = sum(1 for x in self.program[:end] if isinstance(x, int))
        m_1 = sum(1 for x in subtree[:sub_start] if isinstance(x, int))
        m_2 = sum(1 for x in subtree[:sub_end] if isinstance(x, int))
        new_pars = self.int_pars[:n_1] + self.int_pars[m_1:m_2] + self.int_pars[n_2:]
        return self.program[:start] + hoist + self.program[end:], new_pars, removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]
        pars = copy(self.int_pars)
        int_count = 0
        for node_index, node in enumerate(program):
            if node_index in mutate:
                if isinstance(node, _Function):
                    arity = node.arity
                    # Find a valid replacement with same arity
                    replacement = len(self.arities[arity])
                    replacement = random_state.randint(replacement)
                    replacement = self.arities[arity][replacement]
                    program[node_index] = replacement
                else:
                    if isinstance(node, int):
                        del pars[int_count]


                    # We've got a terminal, add a const or variable
                    if self.const_range is not None:
                        terminal = random_state.randint(self.n_features + 1)
                    else:
                        terminal = random_state.randint(self.n_features)
                    if terminal == self.n_features:
                        terminal = random_state.uniform(*self.const_range)

                        if self.const_range is None:
                            # We should never get here
                            raise ValueError('A constant was produced with '
                                             'const_range=None.')
                    else:
                        pars.insert(int_count, 1.)
                        int_count += 1

                    program[node_index] = terminal
            elif isinstance(node, int):
                int_count += 1

        return program, pars, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
