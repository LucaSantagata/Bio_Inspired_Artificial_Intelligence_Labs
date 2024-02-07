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
from utils.utils_10.network import NN
from pylab import *

from  utils.utils_10.plot_utils import *

from inspyred import ec
from utils.utils_10.inspyred_utils import NumpyRandomWrapper

from utils.utils_10.car import *
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

def eval(prey, predator, map, config, render=False):
    sensors = 0 if not config["sensors"] else 4
    nrInputNodes = 2 + sensors  # nrIRSensors + nrDistanceSensor + nrBearingSensor
    nrHiddenNodes = int(config["nrHiddenNodes"])
    nrHiddenLayers = int(config["nrHiddenLayers"])
    nrOutputNodes = 5  # 2

    preyAgents = [NN([nrInputNodes, *[nrHiddenNodes for i in range(nrHiddenLayers)],
              nrOutputNodes]) for i in range(len(prey))]
    predatorAgents = [NN([nrInputNodes, *[nrHiddenNodes for i in range(nrHiddenLayers)],
              nrOutputNodes]) for i in range(len(predator))]
    for i in range(len(prey)):
        preyAgents[i].set_weights(prey[i])
    for i in range(len(predator)):
        predatorAgents[i].set_weights(predator[i])

    dis, obsPrey, obsPredator = run_simulationCoevolution(preyAgents, predatorAgents, map=map, render=render)

    return dis, obsPrey, obsPredator


