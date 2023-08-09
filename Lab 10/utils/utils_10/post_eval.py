# !/usr/bin/python
import sys
import pickle 
import os 
from utils.utils_9.exercise_maze import *

if __name__ =="__main__":
    fname = sys.argv[1]
    inputs = pickle.load(open(fname, "rb"))
    eval(inputs[0], inputs[1], inputs[2], True)
    