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
from itertools import product

class SortingNetwork(list):
    """Sorting network class.
    
    From Wikipedia : A sorting network is an abstract mathematical model
    of a network of wires and comparator modules that is used to sort a
    sequence of numbers. Each comparator connects two wires and sort the
    values by outputting the smaller value to one wire, and a larger
    value to the other.
    """
    def __init__(self, dimension, connectors = []):
        self.dimension = dimension
        for wire1, wire2 in connectors:
            self.addConnector(wire1, wire2)
    
    def addConnector(self, wire1, wire2):
        """Add a connector between wire1 and wire2 in the network."""
        if wire1 == wire2:
            return
        
        if wire1 > wire2:
            wire1, wire2 = wire2, wire1

        index = 0
        for level in reversed(self):
            if self.checkConflict(level, wire1, wire2):
                break
            index -= 1
        
        if index == 0:
            self.append([(wire1, wire2)])
        else:
            self[index].append((wire1, wire2))

    def checkConflict(self, level, wire1, wire2):
        """Check if a connection between `wire1` and `wire2` can be 
        added on this `level`."""
        for wires in level:
            if wires[1] >= wire1 and wires[0] <= wire2:
                return True
    
    def sort(self, values):
        """Sort the values in-place based on the connectors in the network."""
        for level in self:
            for wire1, wire2 in level:
                if values[wire1] > values[wire2]:
                    values[wire1], values[wire2] = values[wire2], values[wire1]
    
    def assess(self, cases=None):
        """Try to sort the **cases** using the network, return the number of
        misses. If **cases** is None, test all possible cases according to
        the network dimensionality.
        """
        if cases is None:
            cases = product((0, 1), repeat=self.dimension)
        
        misses = 0
        ordered = [[0]*(self.dimension-i) + [1]*i for i in range(self.dimension+1)]
        for sequence in cases:
            sequence = list(sequence)
            self.sort(sequence)
            misses += (sequence != ordered[sum(sequence)])
        return misses
    
    def draw(self):
        """Return an ASCII representation of the network."""
        str_wires = [["-"]*7 * self.depth]
        str_wires[0][0] = "0"
        str_wires[0][1] = " o"
        str_spaces = []

        for i in range(1, self.dimension):
            str_wires.append(["-"]*7 * self.depth)
            str_spaces.append([" "]*7 * self.depth)
            str_wires[i][0] = str(i)
            str_wires[i][1] = " o"
        
        for index, level in enumerate(self):
            for wire1, wire2 in level:
                str_wires[wire1][(index+1)*6] = "x"
                str_wires[wire2][(index+1)*6] = "x"
                for i in range(wire1, wire2):
                    str_spaces[i][(index+1)*6+1] = "|"
                for i in range(wire1+1, wire2):
                    str_wires[i][(index+1)*6] = "|"
        
        network_draw = "".join(str_wires[0])
        for line, space in zip(str_wires[1:], str_spaces):
            network_draw += "\n"
            network_draw += "".join(space)
            network_draw += "\n"
            network_draw += "".join(line)
        return network_draw
    
    @property
    def depth(self):
        """Return the number of parallel steps that it takes to sort any input.
        """
        return len(self)
    
    @property
    def length(self):
        """Return the number of comparison-swap used."""
        return sum(len(level) for level in self)


def evalNetwork(host, parasite, dimension):
    network = SortingNetwork(dimension, host)
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



def main(seed, creator, htoolbox, ptoolbox, config):
    INPUTS= config["INPUTS"]                  # length of the input sequence to sort
    POP_SIZE_HOSTS = config["POP_SIZE_HOSTS"]        # population size for hsots
    POP_SIZE_PARASITES = config["POP_SIZE_PARASITES"]    # population size for parasites
    HOF_SIZE = config["HOF_SIZE"]                # size of the Hall-of-Fame
    MAXGEN = config["MAXGEN"]               # number of generations
    H_CXPB = config["H_CXPB"]
    H_MUTPB = config["H_MUTPB"]  # crossover and mutation probability for hosts
    P_CXPB=config["P_CXPB"]
    P_MUTPB = config["P_MUTPB"] # crossover and mutation probability for parasites
    H_TRNMT_SIZE = config["H_TRNMT_SIZE"]            # tournament size for hosts
    P_TRNMT_SIZE = config["P_TRNMT_SIZE"]            # tournament size for parasites
    P_NUM_SEQ = config["P_NUM_SEQ"]              # number of shuffled sequences for each parasite
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
    
    best_network = SortingNetwork(INPUTS, hof[0])
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