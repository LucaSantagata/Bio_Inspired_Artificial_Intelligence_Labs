# !/usr/bin/python
from utils.utils_9.exercise_maze import *
import os 

# --------------------------------------------------------------------------- #
# Change this part, but not the map

def fitness_eval(distanceToTarget, pathLength, noOfTimestepsWithCollisions,
                 timestepToReachTarget, timestepsOnTarget):
    fitness = distanceToTarget
    return fitness


if __name__=="__main__":
    config = {"sensors": False,
                "nrHiddenNodes":  5,
                "nrHiddenLayers": 1,
              "map":"white.png"
             }
    seed = 0
    rng = NumpyRandomWrapper(seed)

    # --------------------------------------------------------------------------- #
    # EA configuration
    display = True

    popSize = 10  # population size
    numGen = 10  # used with generation_termination
    numEval = 2500  # used with evaluation_termination
    tournamentSize = 2  # tournament size (default 2)
    mutationRate = 0.2  # mutation rate, per gene (default 0.1)
    gaussianMean = 0  #  mean of the Gaussian distribution used for mutation
    gaussianStdev = 0.1  #  std. dev. of the Gaussian distribution used for mutation
    crossoverRate = 1.0  # rate at which crossover is performed (default 1.0)
    numCrossoverPoints = 1  # number of crossover points used (default 1)
    selectionSize = popSize  # selection size (i.e. how many individuals are selected for reproduction)
    numElites = 1  # no. of elites (i.e. best individuals that are kept in the population)

    # the evolutionary algorithm (EvolutionaryComputation is a fully configurable evolutionary algorithm)
    #  standard GA, ES, SA, DE, EDA, PAES, NSGA2, PSO and ACO are also available
    ea = inspyred.ec.EvolutionaryComputation(rng)

    # observers: provide various logging features
    if display:
        ea.observer = [inspyred.ec.observers.stats_observer,
                       plot_observer]
        # inspyred.ec.observers.file_observer,
        # inspyred.ec.observers.best_observer,
        # inspyred.ec.observers.population_observer,

    #  selection operator
    # ea.selector = inspyred.ec.selectors.truncation_selection
    # ea.selector = inspyred.ec.selectors.uniform_selection
    # ea.selector = inspyred.ec.selectors.fitness_proportionate_selection
    # ea.selector = inspyred.ec.selectors.rank_selection
    ea.selector = inspyred.ec.selectors.tournament_selection

    # variation operators (mutation/crossover)
    ea.variator = [inspyred.ec.variators.gaussian_mutation,
                   inspyred.ec.variators.n_point_crossover]
    # inspyred.ec.variators.random_reset_mutation,
    # inspyred.ec.variators.inversion_mutation,
    # inspyred.ec.variators.uniform_crossover,
    # inspyred.ec.variators.partially_matched_crossover,

    # replacement operator
    # ea.replacer = inspyred.ec.replacers.truncation_replacement
    # ea.replacer = inspyred.ec.replacers.steady_state_replacement
    # ea.replacer = inspyred.ec.replacers.random_replacement
    # ea.replacer = inspyred.ec.replacers.plus_replacement
    # ea.replacer = inspyred.ec.replacers.comma_replacement
    # ea.replacer = inspyred.ec.replacers.crowding_replacement
    # ea.replacer = inspyred.ec.replacers.simulated_annealing_replacement
    # ea.replacer = inspyred.ec.replacers.nsga_replacement
    # ea.replacer = inspyred.ec.replacers.paes_replacement
    ea.replacer = inspyred.ec.replacers.generational_replacement

    # termination condition
    # ea.terminator = inspyred.ec.terminators.evaluation_termination
    # ea.terminator = inspyred.ec.terminators.no_improvement_termination
    # ea.terminator = inspyred.ec.terminators.diversity_termination
    # ea.terminator = inspyred.ec.terminators.time_termination
    ea.terminator = inspyred.ec.terminators.generation_termination

    # --------------------------------------------------------------------------- #

    # the robot maze navigation problem
    problem = RobotEvaluator(config, seed, eval_func=fitness_eval, maximize=False)

    args = {}
    args["fig_title"] = "EA"

    # run the EA
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          bounder=problem.bounder,
                          maximize=problem.maximize,
                          pop_size=popSize,
                          max_generations=numGen,
                          # max_evaluations=numEval,
                          tournament_size=tournamentSize,
                          mutation_rate=mutationRate,
                          gaussian_mean=gaussianMean,
                          gaussian_stdev=gaussianStdev,
                          crossover_rate=crossoverRate,
                          num_crossover_points=numCrossoverPoints,
                          num_selected=selectionSize,
                          num_elites=numElites, **args)

    # --------------------------------------------------------------------------- #
    best_candidate = final_pop[0].candidate
    best_fitness = final_pop[0].fitness
    os.makedirs("results", exist_ok=True)
    pickle.dump(([best_candidate], "utils/utils_9/"+config["map"], config, True), open("results/best_"+str(seed)+".pkl", "wb"))