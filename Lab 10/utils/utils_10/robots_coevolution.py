# -*- coding: utf-8 -*-

from random import shuffle

import inspyred

from pylab import *

from inspyred import ec
from utils.utils_10.exercise_maze import *
from utils.utils_10.network import *


class ArchiveSolutions():

    def __init__(self, size=None):
        self.candidates = []
        self.fitnesses = []
        self.size = size

    def appendToArchive(self, candidate, fitness=None, maximize=None, archiveType=None):
        if self.size is None or \
                archiveType is None or \
                archiveType == "HALLOFFAME" or \
                (len(self.candidates) < self.size and len(self.fitnesses) < self.size):
            if candidate in self.candidates:
                index = self.candidates.index(candidate)
                self.fitnesses[index] = fitness
            else:
                self.candidates.append(candidate)
                self.fitnesses.append(fitness)
        else:
            if archiveType == "GENERATION":
                # delete the oldest candidate and add the new one
                if candidate in self.candidates:
                    index = self.candidates.index(candidate)
                    self.fitnesses[index] = fitness
                else:
                    del self.candidates[0]
                    del self.fitnesses[0]
                    self.candidates.append(candidate)
                    self.fitnesses.append(fitness)
            elif archiveType == "BEST":
                # find worst candidate in the archive
                if maximize:
                    worstFitness = min(self.fitnesses)
                else:
                    worstFitness = max(self.fitnesses)
                worstIndex = self.fitnesses.index(worstFitness)
                #  replace it if the new candidate is better than the worst candidate in the archive
                if (fitness > worstFitness and maximize) or \
                        (fitness < worstFitness and not maximize):
                    if candidate in self.candidates:
                        index = self.candidates.index(candidate)
                        self.fitnesses[index] = fitness
                    else:
                        del self.candidates[worstIndex]
                        del self.fitnesses[worstIndex]
                        self.candidates.append(candidate)
                        self.fitnesses.append(fitness)

    def getIndexesOfOpponents(self, numOpponents):
        archiveSize = len(self.candidates)
        indexes = list(range(archiveSize))
        shuffle(indexes)
        numOpponents = min(numOpponents, archiveSize)
        return indexes[0:numOpponents]


# --------------------------------------------------------------------------- #
#  Util functions

def readConfigFile(file):
    myvars = {}
    with open(file) as f:
        lines = f.read().splitlines()
        for line in lines:
            if line.startswith("#"):
                pass
            else:
                if "=" in line:
                    name, var = line.partition("=")[::2]
                    myvars[name.strip()] = var.strip()
    return myvars


def writeCandidatesToFile(file, candidates):
    with open(file, "w") as f:
        for candidate in candidates:
            for i in np.arange(len(candidate) - 1):
                f.write(str(candidate[i]) + " ")
            f.write(str(candidate[i]) + "\n")


def readFileAsMatrix(file):
    with open(file) as f:
        lines = f.read().splitlines()
        matrix = []
        for line in lines:
            row = []
            for value in line.split():
                row.append(float(value.replace(",", ".")))
            matrix.append(row)
        return matrix


def getAggregateFitness(fitness_tmp, maximize, archiveUpdate):
    if archiveUpdate == "AVERAGE":
        fitness = np.mean(fitness_tmp)
    elif archiveUpdate == "WORST":
        if maximize:
            fitness = np.min(fitness_tmp)
        else:
            fitness = np.max(fitness_tmp)

    return fitness


def getIndexOfBest(fitnesses, maximize):
    if maximize:
        bestFitness = max(fitnesses)
    else:
        bestFitness = min(fitnesses)
    bestIndex = fitnesses.index(bestFitness)
    return bestIndex


# --------------------------------------------------------------------------- #
#  The robot evaluator class

class RobotEvaluator():

    def __init__(self, config, eval_func_prey, eval_func_predator, name, q_mine, q_his, seed, maximize):

        self.config = config
        self.name = name
        self.q_mine = q_mine
        self.q_his = q_his
        self.seed = seed
        self.shared_variables = config["shared_variables"]
        sensors = 0 if not config["sensors"] else 4
        nrInputNodes = 2 + sensors  # nrIRSensors + nrDistanceSensor + nrBearingSensor
        nrHiddenNodes = int(config["nrHiddenNodes"])
        nrHiddenLayers = int(config["nrHiddenLayers"])
        nrOutputNodes = 5  # 2

        # calculate the no. of weights
        fka = NN([nrInputNodes, *[nrHiddenNodes for i in range(nrHiddenLayers)], nrOutputNodes])

        nrWeights = fka.nweights

        self.geneMin = -3.  # float(parameters["geneMin"])
        self.geneMax = 3.  # float(parameters["geneMax"])
        self.nrTimeStepsGen = 0  # int(parameters["nrTimeStepsGen"])
        self.fitness_evaluator_prey = eval_func_prey
        self.fitness_evaluator_predator = eval_func_predator

        self.nrWeights = nrWeights
        self.seed = seed
        self.bounder = ec.Bounder([self.geneMin] * self.nrWeights,
                                  [self.geneMax] * self.nrWeights)
        self.maximize = maximize

        self.genCount = 0
        self.archiveType = config["archiveType"]
        self.numOpponents = config["numOpponents"]
        self.updateBothArchives = config["updateBothArchives"]
        self.showArchives = config["showArchives"]
        self.numGen = config["numGen"]
        self.archiveUpdate = config["archiveUpdate"]

    def generator(self, random, args):
        return [random.uniform(self.geneMin, self.geneMax) for _ in range(self.nrWeights)]

    def _avg_distance(self, obs):
        tmp = []
        for i in range(len(obs)):
            tmp.append(obs[i][0])

        return np.mean(tmp)

    def _best_distance(self, obs, maximize):
        distance = [obs[i][0] for i in range(len(obs))]
        if maximize:
            return max(distance)
        else:
            return min(distance)

    def _first_contact(self, obs):
        t = 250
        distance = [obs[i][0] for i in range(len(obs))]
        for i in range(len(distance)):
            if distance[i] <= 15.:
                return i
        return t

    def evaluator(self, candidates, args):

        # get lock
        self.q_mine.get()

        #  identify candidates and opponents
        if self.genCount == 0:
            # at the first generation, let all preys compete against all predators
            preys = self.shared_variables.initialPreys.candidates
            predators = self.shared_variables.initialPredators.candidates
        else:
            #  at the next generations, let all preys (predators) compete against individuals in the archives of predators (preys)
            if self.name == "Preys":
                preys = candidates
                if self.archiveType == "HALLOFFAME":
                    indexesOfOpponents = self.shared_variables.archivePredators.getIndexesOfOpponents(self.numOpponents)
                    predators = []
                    archiveSize = len(self.shared_variables.archivePredators.candidates)
                    for i in range(min(self.numOpponents, archiveSize)):
                        predators.append(self.shared_variables.archivePredators.candidates[indexesOfOpponents[i]])
                else:
                    predators = self.shared_variables.archivePredators.candidates
            elif self.name == "Predators":
                predators = candidates
                if self.archiveType == "HALLOFFAME":
                    indexesOfOpponents = self.shared_variables.archivePreys.getIndexesOfOpponents(self.numOpponents)
                    preys = []
                    archiveSize = len(self.shared_variables.archivePreys.candidates)
                    for i in range(min(self.numOpponents, archiveSize)):
                        preys.append(self.shared_variables.archivePreys.candidates[indexesOfOpponents[i]])
                else:
                    preys = self.shared_variables.archivePreys.candidates

        #  create the candidate populations to evaluate
        # we assume that the population is split in two halves, one for preys and one for predators
        """
            n (preys) repeated m (predators) times
            vs
            1 predator -repeated n (preys)- repeated m (predators) times

            [                   [               -
            prey_1              predator_1      |
            prey_2              predator_1      |
            ...                 ...             n
            prey_n              predator_1      |
            ]                   ]               -

            [                   [
            prey_1              predator_2
            prey_2              predator_2
            ...                 ...
            prey_n              predator_2
            ]                   ]

            ...                 ...

            [                   [
            prey_1              predator_m
            prey_2              predator_m
            ...                 ...
            prey_n              predator_m
            ]                   ]
        """
        repeatedPreys = []
        repreatedPredator = []
        # append preys
        for prey in preys:
            for predator in predators:
                repeatedPreys.append(prey)
                repreatedPredator.append(predator)

        # run the simulator
        dis, obsP, obsPr = eval(repeatedPreys, repreatedPredator, "utils/utils_10/" + self.config["map"], self.config,
                                False)
        numRobots = len(dis)
        # TODO: calculate fitness here
        fitnessTmpPrey = []
        fitnessTmpPredator = []

        for i in np.arange(numRobots):
            finalDistanceToTarget = dis[i]  #
            avgDistance = self._avg_distance(obsP[i])
            minDistanceToTarget = self._best_distance(obsP[i], False)
            maxDistanceToTarget = self._best_distance(obsP[i], True)
            timeToContact = self._first_contact(obsP[i])
            fitnessPrey = self.fitness_evaluator_prey(finalDistanceToTarget,avgDistance, minDistanceToTarget, maxDistanceToTarget,
                                                      timeToContact)
            fitnessPredator = self.fitness_evaluator_predator(finalDistanceToTarget,avgDistance, minDistanceToTarget,
                                                              maxDistanceToTarget,
                                                              timeToContact)

            fitnessTmpPrey.append(fitnessPrey)
            fitnessTmpPredator.append(fitnessPredator)

            """
            if i < numRobots/2:
                # preys
                fitnessTmp.append(minDistanceToTarget)
            else:
                # predators
                fitnessTmp.append(timeToContact)
            """

        # update fitness and archives
        fitness_preys = []
        fitness_predators = []

        numPredators = len(predators)
        numPreys = len(preys)
        # print(fitnessTmp)
        # --------------------------------------------------------------------------- #
        if self.updateBothArchives:
            # (update alternative) in this case at each step we update both archives
            # update fitness of preys
            for i in range(numPreys):
                prey = preys[i]
                indexes = np.arange(i * numPredators, (i + 1) * numPredators)
                fitness_prey = getAggregateFitness(np.array(fitnessTmpPrey)[indexes],
                                                   self.shared_variables.problemPreysMaximize, self.archiveUpdate)
                fitness_preys.append(fitness_prey)
            if self.archiveType == "GENERATION" or self.archiveType == "HALLOFFAME":
                # get best prey in the current population
                indexOfBestPrey = getIndexOfBest(fitness_preys, self.shared_variables.problemPreysMaximize)
                bestPrey = preys[indexOfBestPrey]
                bestPreyFitness = fitness_preys[indexOfBestPrey]
                # update archive of preys
                self.shared_variables.archivePreys.appendToArchive(bestPrey, bestPreyFitness,
                                                             self.shared_variables.problemPreysMaximize, self.archiveType)
            elif self.archiveType == "BEST":
                # update archive of preys
                for i in range(numPreys):
                    prey = preys[i]
                    fitness_prey = fitness_preys[i]
                    self.shared_variables.archivePreys.appendToArchive(prey, fitness_prey,
                                                                 self.shared_variables.problemPreysMaximize, self.archiveType)

            # update fitness of predators
            for i in range(numPredators):
                predator = predators[i]
                indexes = np.arange(i, numPreys * numPredators, numPredators)
                fitness_predator = getAggregateFitness(np.array(fitnessTmpPredator)[indexes],
                                                       self.shared_variables.problemPredatorsMaximize, self.archiveUpdate)
                fitness_predators.append(fitness_predator)
            if self.archiveType == "GENERATION" or self.archiveType == "HALLOFFAME":
                # get best predator in the current population
                indexOfBestPredator = getIndexOfBest(fitness_predators, self.shared_variables.problemPredatorsMaximize)
                bestPredator = predators[indexOfBestPredator]
                bestPredatorFitness = fitness_predators[indexOfBestPredator]
                # update archive of predators
                self.shared_variables.archivePredators.appendToArchive(bestPredator, bestPredatorFitness,
                                                                 self.shared_variables.problemPredatorsMaximize,
                                                                 self.archiveType)
            elif self.archiveType == "BEST":
                # update archive of predators
                for i in range(numPredators):
                    predator = predators[i]
                    fitness_predator = fitness_predators[i]
                    self.shared_variables.archivePredators.appendToArchive(predator, fitness_predator,
                                                                     self.shared_variables.problemPredatorsMaximize,
                                                                     self.archiveType)
                    ############################# here
        else:
            # (update alternative) in this case at each step we update only one archive
            if self.name == "Preys":
                # update fitness of preys
                for i in range(numPreys):
                    prey = preys[i]
                    indexes = np.arange(i * numPredators, (i + 1) * numPredators)
                    fitness_prey = getAggregateFitness(np.array(fitnessTmpPrey)[indexes],
                                                       self.shared_variables.problemPreysMaximize, self.archiveUpdate)
                    fitness_preys.append(fitness_prey)
                if self.archiveType == "GENERATION" or self.archiveType == "HALLOFFAME":
                    # get best prey in the current population
                    indexOfBestPrey = getIndexOfBest(fitness_preys, self.shared_variables.problemPreysMaximize)
                    bestPrey = preys[indexOfBestPrey]
                    bestPreyFitness = fitness_preys[indexOfBestPrey]
                    # update archive of preys
                    self.shared_variables.archivePreys.appendToArchive(bestPrey, bestPreyFitness,
                                                                 self.shared_variables.problemPreysMaximize, self.archiveType)
                elif self.archiveType == "BEST":
                    # update archive of preys
                    for i in range(numPreys):
                        prey = preys[i]
                        fitness_prey = fitness_preys[i]
                        self.shared_variables.archivePreys.appendToArchive(prey, fitness_prey,
                                                                     self.shared_variables.problemPreysMaximize,
                                                                     self.archiveType)
            elif self.name == "Predators":
                # update fitness of predators
                for i in range(numPredators):
                    predator = predators[i]
                    indexes = np.arange(i, numPreys * numPredators, numPredators)
                    fitness_predator = getAggregateFitness(np.array(fitnessTmpPredator)[indexes],
                                                           self.shared_variables.problemPredatorsMaximize, self.archiveUpdate)
                    fitness_predators.append(fitness_predator)
                if self.archiveType == "GENERATION" or self.archiveType == "HALLOFFAME":
                    # get best predator in the current population
                    indexOfBestPredator = getIndexOfBest(fitness_predators, self.shared_variables.problemPredatorsMaximize)
                    bestPredator = predators[indexOfBestPredator]
                    bestPredatorFitness = fitness_predators[indexOfBestPredator]
                    # update archive of predators
                    self.shared_variables.archivePredators.appendToArchive(bestPredator, bestPredatorFitness,
                                                                     self.shared_variables.problemPredatorsMaximize,
                                                                     self.archiveType)
                elif self.archiveType == "BEST":
                    # update archive of predators
                    for i in range(numPredators):
                        predator = predators[i]
                        fitness_predator = fitness_predators[i]
                        self.shared_variables.archivePredators.appendToArchive(predator, fitness_predator,
                                                                         self.shared_variables.problemPredatorsMaximize,
                                                                         self.archiveType)

        # --------------------------------------------------------------------------- #

        if self.name == "Preys":
            fitness = fitness_preys
        elif self.name == "Predators":
            fitness = fitness_predators

        # show archives
        if self.showArchives:
            archive = "Archive preys: [ "
            for x in self.shared_variables.archivePreys.fitnesses:
                archive += "{:.4f}".format(x) + " "
            print(archive + "]")

            archive = "Archive predators: [ "
            for x in self.shared_variables.archivePredators.fitnesses:
                archive += "{:.4f}".format(x) + " "
            print(archive + "]")

        print(self.name, self.genCount, "/", self.numGen)

        # increment generation counter
        self.genCount += 1

        # release lock
        self.q_his.put(1)

        return fitness


# --------------------------------------------------------------------------- #

def runEA(problem, display, rng, config):
    # --------------------------------------------------------------------------- #
    # EA configuration

    # the evolutionary algorithm (EvolutionaryComputation is a fully configurable evolutionary algorithm)
    #  standard GA, ES, SA, DE, EDA, PAES, NSGA2, PSO and ACO are also available
    shared_variable = config["shared_variables"]
    ea = inspyred.ec.EvolutionaryComputation(rng)

    # observers: provide various logging features
    if display:
        ea.observer = [inspyred.ec.observers.file_observer]
        # inspyred.ec.observers.stats_observer
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

    if problem.name == "Preys":
        initialPopulation = shared_variable.initialPreys.candidates
    elif problem.name == "Predators":
        initialPopulation = shared_variable.initialPredators.candidates

    # run the EA
    final_pop = ea.evolve(seeds=initialPopulation,
                          generator=problem.generator,
                          evaluator=problem.evaluator,
                          bounder=problem.bounder,
                          maximize=problem.maximize,
                          pop_size=config['popSize'],
                          max_generations=config['numGen'],
                          # max_evaluations=config['numEval'],
                          tournament_size=config['tournamentSize'],
                          mutation_rate=config['mutationRate'],
                          gaussian_mean=config['gaussianMean'],
                          gaussian_stdev=config['gaussianStdev'],
                          crossover_rate=config['crossoverRate'],
                          num_crossover_points=config['numCrossoverPoints'],
                          num_selected=config['selectionSize'],
                          num_elites=config['numElites'],
                          statistics_file=open("stats_" + problem.name + ".csv", "w"),
                          individuals_file=open("individuals_" + problem.name + ".csv", "w"))

    # --------------------------------------------------------------------------- #

    return final_pop
