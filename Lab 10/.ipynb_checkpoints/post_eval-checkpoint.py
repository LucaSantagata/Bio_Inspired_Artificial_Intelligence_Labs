# !/usr/bin/python
import sys
import pickle
import os
from utils.utils_10.exercise_maze import *

if __name__ == "__main__":
    fname = sys.argv[1]
    temp = pickle.load(open(fname, "rb"))
    best = temp[0]
    config = temp[1]
    dis, _, _ = eval([best[0]], [best[1]], map="utils/utils_10/white.png", config=config, render=True)
    print(dis)
