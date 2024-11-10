import operator
import random
import numpy as np
import math
import functools
from deap import base, creator, gp, tools, algorithms
import multiprocessing

def protectedDiv(left, right):
    return left / right if right != 0 else 1

def protectedLog(x):
    return math.log(x) if x > 1e-3 else 0

def protectedExp(x):
    return math.exp(x) if x < 700 else float('inf')  # to prevent overflow

class SymbolicRegressor:
    '''NOTE the default hyperparameters are best hyperparameters from https://arxiv.org/pdf/1912.04871'''
    def __init__(
        self, 
        dataset, # numpy array with shape (n_samples, n_features), last column assumed to be target
        tournament_size=2, # tournament size for selection
        # population_size=1000, # population size
        population_size=500, # population size
        # population_size=24, # population size
        # generations=40, # number of generation
        # generations=2, # number of generation
        # generations=20, # number of generation
        generations=10, # number of generation
        cxpb=0.95, # crossover probability
        mutpb=0.05, # mutation probability
        ):
        self.dataset = dataset
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.generations = generations
        self.cxpb = cxpb
        self.mutpb = mutpb

        num_inputs = dataset.shape[1] - 1 # number of arguments, last column is the target

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.pset = gp.PrimitiveSet("MAIN", num_inputs) # number of arguments
        # Koza library set
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(protectedDiv, 2)
        # self.pset.addPrimitive(np.sin, 1)
        # self.pset.addPrimitive(np.cos, 1)
        # self.pset.addPrimitive(protectedExp, 1)
        # self.pset.addPrimitive(protectedLog, 1)
        
        # Add ephemeral constants
        # self.pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1)) # constants randomly generated between -1 and 1
        self.pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -1, 1))

        arg_names = {f'ARG{i}': f'x{i}' for i in range(num_inputs)} # rename arguments
        self.pset.renameArguments(**arg_names)

        # toolbox of genetic operators
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genFull, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.evalSymbReg)
        self.toolbox.register("select", tools.selTournament, tournsize=tournament_size) # tournament size 3
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def evalSymbReg(self, individual, dataset=None):
        # mean squared error
        if dataset is None:
            dataset = self.dataset
        X = dataset[:, :-1]
        Y = dataset[:, -1]
        func = self.toolbox.compile(expr=individual)
        # sqerrors = ((func(x) - y)**2 for x, y in self.dataset)
        sqerrors = ((func(*x) - y)**2 for x, y in zip(X, Y))
        return math.fsum(sqerrors) / len(self.dataset),

    def optimize(self):
        # cpu_count = multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(cpu_count)
        # self.toolbox.register("map", pool.map)
        population_size = self.population_size
        generations = self.generations
        cxpb = self.cxpb
        mutpb = self.mutpb

        # initialize population and hall of fame
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1) # track n=1 best individuals

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb, mutpb, generations, stats=mstats,
                                       halloffame=hof, verbose=True)

        self.best_individual = hof[0]
        # pool.close()
        # pool.join()
        return pop, log, hof

# Example symbolic regression discovery function (placeholder)
def discover_equations(X, y):
    # Placeholder implementation of symbolic regression for single variable
    # You would replace this with the actual symbolic regression implementation
    # For demonstration, we just print the shapes of X and y
    print("Discovering equations with X shape:", X.shape, "and y shape:", y.shape)
    # Return dummy equations for demonstration
    dataset = np.column_stack((X, y))

    regressor = SymbolicRegressor(dataset)
    regressor.optimize()
    print("Best individual is: ", regressor.best_individual, "| Fitness (MSE): ", regressor.best_individual.fitness.values[0])
    return regressor

# Adapted function for multi-variate search
def discover_multivariate_equations_and_evaluate(ode_trajs_train, ode_trajs_test, state_dim=None, logger=None, env=None, config={}):
    y = np.diff(ode_trajs_train, axis=1)
    y_test = np.diff(ode_trajs_test, axis=1)
    X = ode_trajs_train[:, :-1, :]
    X_test = ode_trajs_test[:, :-1, :]
    assert (ode_trajs_train[:,:-1,:] + y == ode_trajs_train[:,1:,:]).any()
    # print("Shape of X:", X.shape)
    # print("Shape of y:", y.shape)
    batch_size, trajectory_length_minus_one, feature_dimensions = X.shape
    X_flat = X.reshape(-1, feature_dimensions)
    X_flat_test = X_test.reshape(-1, feature_dimensions)
    y_flat = y.reshape(-1, feature_dimensions)
    y_flat_test = y_test.reshape(-1, feature_dimensions)
    # print("Shape of flattened X:", X_flat.shape)
    # print("Shape of flattened y:", y_flat.shape)
    num_variables = y_flat.shape[1]
    regressors = []
    errors = []
    
    for i in range(state_dim):
        y_flat_i = y_flat[:, i]
        y_flat_test_i = y_flat_test[:, i]
        regressor = discover_equations(X_flat, y_flat_i)
        func = regressor.toolbox.compile(expr=regressor.best_individual)
        y_pred = np.array([func(*x) for x in X_flat_test]).reshape(batch_size, -1)
        test_mse = np.mean(((ode_trajs_test[:,:-1,i] + y_pred) - ode_trajs_test[:,1:,i])**2)
        deap_mse = regressor.evalSymbReg(regressor.best_individual, dataset=np.column_stack((X_flat_test, y_flat_test_i)))
        logger.info(f'Test MSE for variable {i + 1}: {test_mse} | DEAP MSE: {deap_mse}')
        errors.append(test_mse)
    test_mse_overall = np.mean(errors)
    logger.info(f'Overall test MSE: {test_mse_overall}')
    return test_mse_overall