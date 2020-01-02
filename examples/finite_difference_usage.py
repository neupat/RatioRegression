from ratio_regression.examples.finite_differenciator import get_ratio, save_div
from ratio_regression.utils import INFINITY_VALUE
import numpy as np
from ratio_regression.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
import sys


pipe = False


#Define an example function and the number of features it takes
n_features = 5

def f(X, **kwargs):
    val = np.sin(X[:, 0]) * X[:, 1]
    return val / X[:, 2] + np.cos(X[:, 3]**2)


sqrt_N = 15
N = sqrt_N**2
number_to_name = {i:'X_'+str(i) for i in range(n_features)}
default_variables = {'X_0':-2., 'X_1':1.3, 'X_2':1.4, 'X_3':0.9, 'X_4':2.3}
ranges = {'X_0':(-3,-1), "X_1":(0,2), 'X_2':(0.3,2.3), 'X_3':(0.3,2.3), 'X_4':(-1,1)}


ratios_to_handle = []
for i in range(n_features):
    ratios_to_handle += [(i,j) for j in range(i+1,n_features)]

for handle in ratios_to_handle:
    print("handling {} vs {}".format(handle[0],handle[1]))
    print("="*30)
    if pipe:
        original = sys.stdout
        file = open('outputs//{}vs{}.txt'.format(handle[0],handle[1]), 'w')
        sys.stdout = file

    print("Handling {} vs {}".format(*[number_to_name[j] for j in handle]))


    #region Preparation

    handle_vals = []

    for i in handle:
        name = number_to_name[i]
        handle_vals.append(np.linspace(ranges[name][0],ranges[name][1],sqrt_N))


    a,b = np.meshgrid(handle_vals[0],handle_vals[1])
    a = a.reshape(-1)
    b = b.reshape(-1)
    handle_vals = iter([a,b])


    X = []
    for i in range(n_features):
        if i in handle:
            X.append(handle_vals.__next__())
        else:
            X.append(np.repeat(default_variables[name], N))
    # for g in [number_to_var[x] for x in range(len(number_to_var))]:
    #     print(g.shape)

    for x in X:
        print(x.shape)

    X_algo = np.vstack(tuple([X[i] for i in range(n_features) if i in handle])).T
    X = np.vstack(tuple([X[i] for i in range(n_features)])).T




    noise_size = 5e-3
    print("Noise: +- ",noise_size)
    y = get_ratio(f, X, handle[0], handle[1])
    y += (np.random.rand(len(y)) - 0.5)*noise_size





    print("Shape X data: ", x.shape)
    print("Shape y data: ",y.shape)

    if np.all(np.isnan(y)) or np.all(y > INFINITY_VALUE - noise_size) or  np.all(y < -INFINITY_VALUE + noise_size):
        print("Most likely denominator = 0 everywhere, f does not depend on X1")
        print("Try switching the variables and check if y == 0 everywhere or increase n (Data Points)")
        print("=" * 50)
        print('\n' * 8)
        continue

    if np.all(y < noise_size):
        print("Most likely ratio = 0 everywhere, f does not depend on X0")
        print("=" * 50)
        print('\n' * 8)
        continue

    if np.any(y == save_div(1,0.)):
        print("Warning")


    #### hyper parameters
    tournament_size = 40
    population_size = 7000
    parsimony = 1e-4
    random_state = np.random.randint(0,1e7)
    generations = 15
    p_crossover = 0.3
    p_subtree_mutation = 0.1
    p_hoist_mutation = 0.1
    p_point_mutation = 0.1
    const_range = (-2,4)
    p_point_replace = 0.05
    p_clean = lambda b: 0.0 * b
    p_const = 0.4
    n_const = 1
    p_vars = 0.0
    n_vars = 0


    function_set = ('mul', 'add', 'sub', 'div', 'sin', 'cos')


    import time
    t = time.time()
    est_gp = SymbolicRegressor(population_size=population_size,
                               generations=generations,tournament_size=tournament_size, stopping_criteria=noise_size,
                               p_crossover=p_crossover, p_subtree_mutation=p_subtree_mutation,
                               p_hoist_mutation=p_hoist_mutation, p_point_mutation=p_point_mutation,
                               max_samples=1., verbose=1, const_range=const_range,p_point_replace=p_point_replace,
                               parsimony_coefficient=parsimony, random_state=random_state, function_set=function_set,
                               p_clean=p_clean, p_const=p_const,n_const=n_const,p_vars=p_vars,n_vars=n_vars,
                               feature_names=['X_' + str(i) for i in handle])
    est_gp.fit(X_algo, y)


    #region Pretty Prints
    print("Time for evaluation: ")
    print(time.time() - t, " seconds")
    print("Dirty Solution: ")
    print(est_gp._program)
    clean = est_gp._program
    clean.cleanup()
    print("Parent:")
    print(est_gp._program.parents)
    print("Solution:")
    print(clean)
    print("=" * 50)
    print("Fitness of Solution: ")
    print(clean.raw_fitness_)
    print("=" * 50)
    print("Length of Solution: ")
    print(len(clean.program))
    print("=" * 50)
    print('\n'*8)


    if pipe:
        print("Exiting...")
        sys.stdout = original
        file.close()

    #endregion
