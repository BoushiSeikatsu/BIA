import sys

# setting path
sys.path.append('../')
import TestFunctions as tf
import EvaluationTracker as et
#import TestFunctions as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from importlib import reload
import random
import time
import matplotlib.animation as animation
import math
import copy

class Solution:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.d = dimension
        self.lower = lower_bound # we will use the same bounds for all parameters
        self.upper = upper_bound
        self.params = np.zeros(self.d) #solution parameters
        self.f = np.inf # objective function evaluation

def generatePopulation(solution : Solution, size):
    population = []
    for _ in range(0, size):
        new_solution = Solution(len(solution.params), solution.lower, solution.upper) # Generate solution
        for i in range(0, len(solution.params)): 
            new_solution.params[i] = random.uniform(solution.lower, solution.upper) # Generate its params
            #print(new_solution.params)
        population.append(new_solution)
    return population

# Evaluate entire population on function func
def evaluateAll(population, func, evaluationTracker : et.EvaluationTracker):
    for solution in population:
        z = func(solution.params)
        evaluationTracker.currentEvalCount += 1
        solution.f = z
    return population

def pickBest(population):
    bestScore = sys.maxsize
    bestSolution = None
    for solution in population:
        if solution.f < bestScore:
            bestScore = solution.f
            bestSolution = solution
    return copy.copy(bestSolution)

def generatePRTVector(solution, prt):
    vector = []
    for i in range(0, len(solution.params)):
        r = random.uniform(0,1)
        tmp = 0
        if r < prt:
            tmp = 1
        vector.append(tmp)
    return vector

def calculateNewPosition(current_solution : Solution, leader : Solution, func, step, prtVector, pathLength, evaluationTracker : et.EvaluationTracker):
    t = step
    new_population = Solution(len(current_solution.params), current_solution.lower, current_solution.upper)
    while t < pathLength:
        for i in range(0, len(current_solution.params)):
            #print(type(prtVector))
            new_population.params[i] = current_solution.params[i] + (leader.params[i] - current_solution.params[i]) * t * prtVector[i]
            newEval = func(new_population.params)
            evaluationTracker.currentEvalCount += 1
            if evaluationTracker.currentEvalCount > evaluationTracker.maxEval: # If max eval count has been reached, finish algorithm
                return None
        if newEval < current_solution.f:
            current_solution.params = new_population.params.copy()
            current_solution.f = newEval
        t += step
    return current_solution

def SOMA(solution, func):
    for i in range(0,len(solution.params)):
        solution.params[i] = random.uniform(solution.lower, solution.upper) # Generate random coordinates for the first time

    # Parameters
    pop_size = 30          # Number of individuals in the population
    M_max = 100          # Maximum number of iterations
    step = 0.11                   # Step size for migration
    path_length = 3               # Path length, the extent each individual can move
    prt = 0.4
    evaluationTracker = et.EvaluationTracker()
    #Initialize Population
    population = generatePopulation(solution=solution, size=pop_size)
    population = evaluateAll(population, func, evaluationTracker)
    leader = pickBest(population)
    progressTracker = []
    #Initial point
    progressTracker.append((leader.params.copy()[0], leader.params.copy()[1], leader.f))
    #Main Migration Loop
    for _ in range(0, M_max):
        # Identify the best individual in the population
        new_popuplation = list(np.copy(population)) # new generation
        leader = pickBest(population)
        if progressTracker[-1][0] != leader.params[0] and progressTracker[-1][1] != leader.params[1]:
            progressTracker.append((leader.params.copy()[0], leader.params.copy()[1], leader.f))
        # For each individual in the population (except the best one)
        for i in range(0, pop_size):
            if new_popuplation[i] == leader:
                continue  # Skip the best individual
            #Generate PRTVector
            prtVector = generatePRTVector(solution, prt)
            new_popuplation[i] = calculateNewPosition(new_popuplation[i], leader, func, step, prtVector, path_length, evaluationTracker)
            if new_popuplation[i] is None: # If max evals have been reached
                return progressTracker
            t = step
            
        population = new_popuplation

    return progressTracker