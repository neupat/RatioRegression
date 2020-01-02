import numpy as np
from ratio_regression.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
import sys
import time



# define the goal function to find, will not be used directly
# takes n input variables, here n=2
def f(x):
    return np.sin(1.6*x[0])/(x[1]**2)


# the ratio derivative function used to produce the input data
def ratio(x):
    nom = 1.6*np.cos(1.6*x[0])/x[1]**2
    denom = -2*np.sin(1.6*x[0])/(x[1]**3)#-x[0]
    return nom/denom


#generate data set on N points, where X.shape = (N,n) and y.shape = (N,)
#here N = 25**2 = 625
x = np.linspace(1,15,25)
y = np.linspace(1,15,25)
x,y = np.meshgrid(x,y)
x = np.ravel(x)
y = np.ravel(y)
X = np.stack((x,y),axis=-1)
y = np.apply_along_axis(ratio,1,X)
print(X.shape)
print(y.shape)


#Only use data points that are not to extreme
thresh = 1000
to_use = np.where(np.abs(y) < thresh)[0]
y = y[to_use]
X = X[to_use,:]

print("Shape X data: ", X.shape)
print("Shape y data: ",y.shape)

#set hyperparameters
tournament_size = 40
population_size = 7000
parsimony = 0.1
generations=15
stopping_criteria = 1e-5
p_crossover = 0.3
p_subtree_mutation = 0.1
p_hoist_mutation=0.1
p_point_mutation=0.1
max_samples=1.
const_range=(-2,4)
p_clean=lambda b: 0.0*b
p_const=0.5
n_const=1
p_vars = 0.0
n_vars = 1
function_set = ('mul','div','add', 'sub','sin','cos')


t = time.time()
est_gp = SymbolicRegressor(population_size=population_size,
                           generations=15,tournament_size=tournament_size, stopping_criteria=0.00000001,
                           p_crossover=0.3, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.1, p_point_mutation=0.1,
                           max_samples=1., verbose=1, const_range=(-2,4),
                           parsimony_coefficient=parsimony, random_state=np.random.randint(0,1e7), function_set=function_set,
                           p_const=p_const, n_const=n_const, p_vars=p_vars, n_vars=n_vars,
                           p_clean=p_clean)
est_gp.fit(X, y)

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



x_plot = np.linspace(10,40,100)
rat_plot = lambda a,b: ratio([a,b])
for x2 in [40]:
    y_plot = rat_plot(x_plot,x2)
    to_use = np.where(np.abs(y_plot) < thresh)[0]
    y_plot = y_plot[to_use]

    x_plot2 = x_plot[to_use]
    plt.plot(x_plot2,y_plot,'o',label="True Solution at X_2 = "+str(x2))
    y_eval = clean.ratio_evaluate(np.vstack((x_plot2,np.ones_like(x_plot2)*x2)).T)
    plt.plot(x_plot2,y_eval,label="Evaluation at X_2 = "+str(x2))


plt.legend()
plt.show()
