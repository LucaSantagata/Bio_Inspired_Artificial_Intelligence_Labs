# -*- coding: utf-8 -*-

from random import shuffle
from queue import Queue

import shutil
import os.path
import inspyred

from pylab import *

from inspyred import ec
from Lab_10.utils.utils_10.inspyred_utils import NumpyRandomWrapper
from utils.utils_10.exercise_maze import *
from utils.utils_10.network import *
from utils.utils_10.robots_coevolution import *
import utils.utils_10.cfg as shared_variable

"""
-------------------------------------------------------------------------
Edit this part to do the exercises
"""



config = {"sensors": True,
          "nrHiddenNodes": 3,
          "nrHiddenLayers": 1,
          "map": "white.png",
          # parameters for standard GA
          "popSize": 15,  # population size
          "numGen": 50,  # used with generation_termination
          "tournamentSize": 2,  # tournament size (default 2)
          "mutationRate": 0.2,  # mutation rate, per gene (default 0.1)
          "gaussianMean": 0,  #  mean of the Gaussian distribution used for mutation
          "gaussianStdev": 0.1,  #  std. dev. of the Gaussian distribution used for mutation
          "crossoverRate": 1.0,  # rate at which crossover is performed (default 1.0)
          "numCrossoverPoints": 1,  # number of crossover points used (default 1)
          # selection size (i.e. how many individuals are selected for reproduction)
          "numElites": 1,
          # no. of elites (i.e. best individuals that are kept in the population # parameters for competitive coevolution

          "numOpponents": 1,  # number of opponents against which each robot competes at each generation
          "archiveType": "BEST",  #  possible types: {GENERATION,HALLOFFAME,BEST}
          "archiveUpdate": "AVERAGE",  # possible types: {WORST,AVERAGE}
          "updateBothArchives": False,# True is each generation should update both archives, False otherwise
            "display" : True,
            "showArchives" : False
          }
config["selectionSize"] = config['popSize']


# 1. Generational competition: the archive is filled with the best individuals from previous n generations (e.g. n=5)
# 2. Hall-of-Fame: each new individual is tested against *all best opponents* obtained so far.
#    NOTE: Using this method, the no. of tournaments increases along generations!
#    However, it is sufficient to test new individuals only against a limited sample of n opponents (e.g. n=10)
# 3. Best competition: the archive is filled with the best n (e.g. n=5) individuals from *all* previous generations

def fitness_eval_prey(finalDistanceToTarget, avgDistanceToTarget, minDistanceToTarget, maxDistanceToTarget, timeToContact):
    fitness = avgDistanceToTarget
    return fitness

def fitness_eval_predator(finalDistanceToTarget, avgDistanceToTarget, minDistanceToTarget, maxDistanceToTarget, timeToContact):
    fitness = avgDistanceToTarget
    return fitness

"""
-------------------------------------------------------------------------
"""

# These two archives keep the best preys and best predators
cc = shared_variable.cfgs()

if config["archiveType"] == "GENERATION" or config["archiveType"] == "BEST":
    cc.archivePreys = ArchiveSolutions( config["numOpponents"])
    cc.archivePredators = ArchiveSolutions( config["numOpponents"])
elif  config["archiveType"] == "HALLOFFAME":
    cc.archivePreys = ArchiveSolutions()
    cc.archivePredators = ArchiveSolutions()

#  the initial popolations (we need to make initialize them externally to initialize the archives)

cc.initialPreys = ArchiveSolutions(config['popSize'])
cc.initialPredators = ArchiveSolutions(config['popSize'])

# TODO: change maximize flag depending on how fitness is defined
cc.problemPreysMaximize = True  # e.g. maximize (final/min) distance from predator or maximize time-to-contact
cc.problemPredatorsMaximize = False  # e.g. minimize (final/min) distance from prey or minimize time-to-contact

config["shared_variables"] = cc

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    seed = 0
    rng = NumpyRandomWrapper(seed)
    #  the following queues allow the two threads to alternate their execution
    qAB = Queue()
    qBA = Queue()

    # create the robot evaluator instances
    problemPreys = RobotEvaluator(config, fitness_eval_prey, fitness_eval_predator, "Preys", qAB, qBA, seed, cc.problemPreysMaximize)
    problemPredators = RobotEvaluator(config, fitness_eval_predator, fitness_eval_predator, "Predators", qBA, qAB, seed, cc.problemPredatorsMaximize)

    # create the initial populations
    for i in np.arange(config['popSize']):
        candidatePrey = [(problemPreys.geneMax - problemPreys.geneMin) * rng.random_sample() + problemPreys.geneMin \
                         for _ in range(problemPreys.nrWeights)]
        cc.initialPreys.appendToArchive(candidatePrey)
    for i in np.arange(config['popSize']):
        candidatePredator = [
            (problemPredators.geneMax - problemPredators.geneMin) * rng.random_sample() + problemPredators.geneMin \
            for _ in range(problemPredators.nrWeights)]
        cc.initialPredators.appendToArchive(candidatePredator)

    t1 = threading.Thread(target=runEA, args=(problemPreys,config["display"], rng, config))
    t2 = threading.Thread(target=runEA, args=(problemPredators,config["display"], rng, config))

    # this is needed to unlock the thread "Preys" first
    qAB.put(1)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    if config["display"]:
        """
        # rerun every prey in the archive against every predator in the archive
        preysPredators = []

        # append preys
        for predator in archivePredators.candidates:
            for prey in archivePreys.candidates:
                preysPredators.append(prey)
        # append predators
        for predator in archivePredators.candidates:
            for prey in archivePreys.candidates:
                preysPredators.append(predator)
        """

        # rerun the best prey in the archive against the best predator in the archive
        preysPredators = []
        indexOfBestPrey = getIndexOfBest(cc.archivePreys.fitnesses, cc.problemPreysMaximize)
        bestPrey = cc.archivePreys.candidates[indexOfBestPrey]
        bestPreyFitness = cc.archivePreys.fitnesses[indexOfBestPrey]
        indexOfBestPredator = getIndexOfBest(cc.archivePredators.fitnesses, cc.problemPredatorsMaximize)
        bestPredator = cc.archivePredators.candidates[indexOfBestPredator]
        bestPredatorFitness = cc.archivePredators.fitnesses[indexOfBestPredator]
        print("prey " + str(bestPreyFitness))
        print("predator " + str(bestPredatorFitness))
        preysPredators.append(bestPrey)
        preysPredators.append(bestPredator)
        import pickle

        with open("s.pkl", "wb") as f:
            print(bestPrey)
            pickle.dump((bestPrey, bestPredator), f)
        with open("tmp.pkl", "wb") as f:
            print(bestPrey)
            pickle.dump([(bestPrey, bestPredator), config], f)


        statsPreys = np.transpose(np.loadtxt(open("./stats_Preys.csv", "r"), delimiter=","))
        statsPredators = np.transpose(np.loadtxt(open("./stats_Predators.csv", "r"), delimiter=","))

        # plot fitness trends of preys and predators
        figure("Preys")
        plot(statsPreys[2], label="Worst")
        plot(statsPreys[3], label="Best")
        plot(statsPreys[4], label="Median")
        plot(statsPreys[5], label="Mean")
        # yscale("log")
        xlabel("Generation")
        ylabel("Fitness")
        legend()

        figure("Predators")
        plot(statsPredators[2], label="Worst")
        plot(statsPredators[3], label="Best")
        plot(statsPredators[4], label="Median")
        plot(statsPredators[5], label="Mean")
        # yscale("log")
        xlabel("Generation")
        ylabel("Fitness")
        legend()
        show()