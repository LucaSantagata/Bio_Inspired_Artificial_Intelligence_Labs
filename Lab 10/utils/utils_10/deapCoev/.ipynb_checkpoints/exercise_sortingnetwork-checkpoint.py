# -*- coding: utf-8 -*-

#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import time
import random
import sys

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

import sortingnetwork as sn

"""
-------------------------------------------------------------------------
Edit this part to do the exercises
"""

INPUTS = 5                  # length of the input sequence to sort
POP_SIZE_HOSTS = 300        # population size for hsots
POP_SIZE_PARASITES = 300    # population size for parasites
HOF_SIZE = 1                # size of the Hall-of-Fame
MAXGEN = 50                 # number of generations
H_CXPB, H_MUTPB = 0.5, 0.3  # crossover and mutation probability for hosts
P_CXPB, P_MUTPB = 0.5, 0.3  # crossover and mutation probability for parasites
H_TRNMT_SIZE = 3            # tournament size for hosts
P_TRNMT_SIZE = 3            # tournament size for parasites
P_NUM_SEQ = 20              # number of shuffled sequences for each parasite

"""
-------------------------------------------------------------------------
"""

#--------------------------------------------------------------------
# Util functions

def evalNetwork(host, parasite, dimension):
    network = sn.SortingNetwork(dimension, host)
    return network.assess(parasite),
    # sort all the sequences contained in the parasites,
    # return the number of misses (sequences not correctly sorted)

def genWire(dimension):
    return (random.randrange(dimension), random.randrange(dimension))

def genNetwork(dimension, min_size, max_size):
    size = random.randint(min_size, max_size)
    return [genWire(dimension) for i in range(size)]

def getParasite(dimension):
    return [random.choice((0, 1)) for i in range(dimension)]

def mutNetwork(individual, dimension, mutpb, addpb, delpb, indpb):
    if random.random() < mutpb:
        for index, elem in enumerate(individual):
            if random.random() < indpb:
                individual[index] = genWire(dimension)
    if random.random() < addpb:
        index = random.randint(0, len(individual))
        individual.insert(index, genWire(dimension))
    if random.random() < delpb:
        index = random.randrange(len(individual))
        del individual[index]
    return individual,

def mutParasite(individual, indmut, indpb):
    for i in individual:
        if random.random() < indpb:
            indmut(i)
    return individual,

def cloneHost(individual):
    """Specialized copy function that will work only on a list of tuples
        with no other member than a fitness.
        """
    clone = individual.__class__(individual)
    clone.fitness.values = individual.fitness.values
    return clone

def cloneParasite(individual):
    """Specialized copy function that will work only on a list of lists
        with no other member than a fitness.
        """
    clone = individual.__class__(list(seq) for seq in individual)
    clone.fitness.values = individual.fitness.values
    return clone

#--------------------------------------------------------------------
# The EA parametrization

# this four lines simply tells DEAP that
# 1. hosts want to minimize (the sorting errors)
# 2. parasites want to maximize (the sorting errors)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Host", list, fitness=creator.FitnessMin)
creator.create("Parasite", list, fitness=creator.FitnessMax)

htoolbox = base.Toolbox()
ptoolbox = base.Toolbox()

# register the initialization operators for hosts
htoolbox.register("network", genNetwork, dimension=INPUTS, min_size=INPUTS, max_size=INPUTS*2)
htoolbox.register("individual", tools.initIterate, creator.Host, htoolbox.network)
htoolbox.register("population", tools.initRepeat, list, htoolbox.individual)

# register the initialization operators for parasites
ptoolbox.register("parasite", getParasite, dimension=INPUTS)
# NOTE: each parasite is actually an array of P_NUM_SEQ shuffled sequences (not just one)
ptoolbox.register("individual", tools.initRepeat, creator.Parasite, ptoolbox.parasite, P_NUM_SEQ)
ptoolbox.register("population", tools.initRepeat, list, ptoolbox.individual)

# register the evaluation/crossover/mutation/selection/clone operators for hosts
# we keep the additional specific parameters as they are
htoolbox.register("evaluate", evalNetwork, dimension=INPUTS)
htoolbox.register("mate", tools.cxTwoPoint)
htoolbox.register("mutate", mutNetwork, dimension=INPUTS, mutpb=0.2, addpb=0.01, delpb=0.01, indpb=0.05)
htoolbox.register("select", tools.selTournament, tournsize=H_TRNMT_SIZE)
htoolbox.register("clone", cloneHost)

# register the crossover/mutation/selection/clone operators for parasites
# note that in this case an evaluation function is not defined explicitly
# (parasite"s fitness is the same as the corresponding host, see below)
# we keep the additional specific parameters as they are
ptoolbox.register("mate", tools.cxTwoPoint)
ptoolbox.register("indMutate", tools.mutFlipBit, indpb=0.05)
ptoolbox.register("mutate", mutParasite, indmut=ptoolbox.indMutate, indpb=0.05)
ptoolbox.register("select", tools.selTournament, tournsize=P_TRNMT_SIZE)
ptoolbox.register("clone", cloneParasite)

#--------------------------------------------------------------------

def main(seed):

    random.seed(seed)
    
    hosts = htoolbox.population(n=POP_SIZE_HOSTS)
    parasites = ptoolbox.population(n=POP_SIZE_PARASITES)
    hof = tools.HallOfFame(HOF_SIZE)
    
    """
    print("Initial hosts:")
    hn = 0
    for h in hosts:
        print(h)
        hn += 1
    print("No. of hosts: ", hn)
    
    print("Initial parasites:")
    pn = 0
    for p in parasites:
        print(p)
        pn += 1
    print("No. of parasites: ", pn)
    """
    
    hstats = tools.Statistics(lambda ind: ind.fitness.values)
    hstats.register("avg", numpy.mean)
    hstats.register("std", numpy.std)
    hstats.register("min", numpy.min)
    hstats.register("max", numpy.max)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    fits = htoolbox.map(htoolbox.evaluate, hosts, parasites)
    for host, parasite, fit in zip(hosts, parasites, fits):
        host.fitness.values = parasite.fitness.values = fit
    
    hof.update(hosts)
    record = hstats.compile(hosts)
    logbook.record(gen=0, evals=len(hosts), **record)
    print(logbook.stream)
    
    for g in range(1, MAXGEN):
        hosts = htoolbox.select(hosts, len(hosts))
        parasites = ptoolbox.select(parasites, len(parasites))
        
        hosts = algorithms.varAnd(hosts, htoolbox, H_CXPB, H_MUTPB)
        parasites = algorithms.varAnd(parasites, ptoolbox, P_CXPB, P_MUTPB)
        
        fits = htoolbox.map(htoolbox.evaluate, hosts, parasites)
        for host, parasite, fit in zip(hosts, parasites, fits):
            # this is where hosts and parasites" fitness are assigned
            host.fitness.values = parasite.fitness.values = fit
        
        hof.update(hosts)
        record = hstats.compile(hosts)
        logbook.record(gen=g, evals=len(hosts), **record)
        print(logbook.stream)
    
    best_network = sn.SortingNetwork(INPUTS, hof[0])
    print(best_network)
    print(best_network.draw())
    print("%i errors" % best_network.assess())
    # test the best network against all possible sequences, according to the network dimensionality

    #--------------------------------------------------------------------
    
    gen = logbook.select("gen")
    fit_min = logbook.select("min")
    fit_max = logbook.select("max")
    fit_avg = logbook.select("avg")
    fit_std = logbook.select("std")
    
    fig = plt.figure("Coevolution")
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(gen, fit_min, label="Min")
    line2 = ax1.plot(gen, fit_max, label="Max")
    line3 = ax1.errorbar(gen, fit_avg, yerr=fit_std, label="Avg")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("(Host) Fitness")
    ax1.set_xlim(0,MAXGEN-1)
    ax1.legend()

    plt.show()
    
    #--------------------------------------------------------------------
    
    return hosts, logbook, hof

if __name__ == "__main__":
    if len(sys.argv) > 1 :
        seed = int(sys.argv[1])
    else :
        seed = int(time.time())

    main(seed)
