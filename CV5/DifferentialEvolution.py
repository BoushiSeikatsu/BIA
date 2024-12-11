import sys

# setting path
sys.path.append('../')
import TestFunctions as tf
import EvaluationTracker as et
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from importlib import reload
import random
import time
import matplotlib.animation as animation
import math

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

def DifferentialEvolution(solution, func):
    for i in range(0,len(solution.params)):
        solution.params[i] = random.uniform(solution.lower, solution.upper) # Generate random coordinates for the first time
    #pop = Generate NP random individuals (you can use the class Solution mentioned in Exercise 1)
    #print(solution.params)
    evaluationTracker = et.EvaluationTracker()
    pop_size = 30
    population = generatePopulation(solution, pop_size) # Get population
    population = evaluateAll(population, func, evaluationTracker=evaluationTracker) # Set evaluations for population
    # Hard coded 
    g = 0
    g_maxim = 200
    F = 0.8
    CR = 0.9
    progressTracker = []
    while g < g_maxim :
        new_popuplation = list(np.copy(population)) # new generation
        for i in range(0, len(population)): # x is also denoted as a target vector
            indices = list(range(len(population)))
            indices.remove(i)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)
            mutation = (population[r1].params - population[r2].params)*F + population[r3].params # mutation vector. TAKE CARE FOR BOUNDARIES!
            #print(mutation)
            mutation = np.clip(mutation, solution.lower, solution.upper)
            u = np.zeros(len(solution.params)) # trial vector
            j_rnd = np.random.randint(0, len(solution.params))
        for j in range(0, len(solution.params)):
            if np.random.uniform() < CR or j == j_rnd:
                u[j] = mutation[j] # at least 1 parameter should be from a mutation vector v
            else:
                u[j] = population[i].params[j]
        #f_u = Evaluate trial vector u
        f_u = func(u)
        evaluationTracker.currentEvalCount += 1
        if evaluationTracker.currentEvalCount > evaluationTracker.maxEval:
            return progressTracker
        if f_u <= population[i].f: # We always accept a solution with the same fitness as a target vector
            new_x = Solution(len(solution.params), solution.lower, solution.upper)
            new_x.params = u
            new_x.f = f_u
            new_popuplation[i] = new_x
            population = new_popuplation
            progressTracker.append((new_x.params.copy()[0], new_x.params.copy()[1], f_u))
        g += 1
    return progressTracker