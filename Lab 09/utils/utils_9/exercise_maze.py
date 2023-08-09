# -*- coding: utf-8 -*-

from random import Random
import time

import sys
import math
import shutil
import os.path
import inspyred
import matplotlib
import numpy as np
import pickle
from utils.utils_9.network import NN
from pylab import *

from  utils.utils_9.plot_utils import *

from inspyred import ec
from utils.utils_9.inspyred_utils import NumpyRandomWrapper

from utils.utils_9.car import *
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


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

def fitness_eval(distanceToTarget, pathLength, noOfTimestepsWithCollisions,
                 timestepToReachTarget, timestepsOnTarget):
    fitness = distanceToTarget
    return fitness

def eval(cs, map, config, render=False):
    sensors = 0 if not config["sensors"] else 4
    nrInputNodes = 2 + sensors  # nrIRSensors + nrDistanceSensor + nrBearingSensor
    nrHiddenNodes = int(config["nrHiddenNodes"])
    nrHiddenLayers = int(config["nrHiddenLayers"])
    nrOutputNodes = 5  # 2

    agents = [NN([nrInputNodes, *[nrHiddenNodes for i in range(nrHiddenLayers)],
              nrOutputNodes]) for i in range(len(cs))]
    for i in range(len(cs)):
        agents[i].set_weights(cs[i])
    dis, obs = run_simulation(agents, map=map, render=render)

    return dis, obs

#  this object is used for single-thread evaluations (only pickleable objects can be used in multi-thread)
class RobotEvaluator():

    def __init__(self, config, seed, eval_func, maximize):
        self.config= config

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
        self.fitness_evaluator = eval_func
        self.nrWeights = nrWeights
        self.seed = seed
        self.bounder = ec.Bounder([self.geneMin] * self.nrWeights,
                                  [self.geneMax] * self.nrWeights)
        self.maximize = maximize

        self.genCount = 0

    def generator(self, random, args):
        return [random.uniform(self.geneMin, self.geneMax) for _ in
                range(self.nrWeights)]

    def _distance(self, p1, p2):
        return math.sqrt(
            (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def _distance(self, p1, p2):
        return math.sqrt(
            (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def calcPathLength(self, obs):
        pathLength = 0
        for i in range(len(obs) - 1):
            pathLength += self._distance(obs[i][:2], obs[i + 1][:2])
        return pathLength

    def calcStepWithCollisions(self, obs, eps=10e-5):
        count = 0
        for i in range(len(obs)):
            if max(obs[i][2:]) <= eps:
                count += 1
        return count

    def timeStepToReachTarget(self, obs):
        count = 250
        for i in range(len(obs)):
            if self._distance(obs[i][:2], [1730, 165]) < 32:
                return i
        return count

    def timeOnTarget(self, obs):
        count = 0
        for i in range(len(obs)):
            if self._distance(obs[i][:2], [1730, 165]) < 32:
                count += 1
        return count

    def evaluator(self, candidates, args):
        times = []
        results = []
        s = time.time()
        results, observations = eval(candidates, "utils/utils_9/"+self.config["map"], self.config, False)
        fitness = []
        for i in np.arange(len(candidates)):
            distanceToTarget = results[i]
            pathLength = self.calcPathLength(observations[i])
            noOfTimestepsWithCollisions = self.calcStepWithCollisions(observations[i])
            timestepToReachTarget = self.timeStepToReachTarget(observations[i])
            timestepsOnTarget = self.timeOnTarget(observations[i])
            fitness_i = self.fitness_evaluator(distanceToTarget, pathLength, noOfTimestepsWithCollisions, timestepToReachTarget, timestepsOnTarget)
            fitness.append(fitness_i)

        self.genCount += 1
        return fitness
