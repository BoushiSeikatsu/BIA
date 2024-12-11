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
import copy
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

def pickBest(population):
    bestScore = sys.maxsize
    bestSolution = None
    for solution in population:
        if solution.f < bestScore:
            bestScore = solution.f
            bestSolution = solution
    return copy.copy(bestSolution)

def generateVectors(population, v_min = -1, v_max = 1):
    vectors = []
    for solution in population:
        vector = []
        for i in range(0, len(solution.params)):
            vector.append(random.uniform(v_min,v_max)) # Vectors parameters will have values from -1 to 1
        vectors.append(vector)
    return vectors

def updateVector(solution : Solution, bestPSolution : Solution, bestGSolution : Solution, vector, index, M_max):
    w_s = 0.9
    w_e = 0.4
    c1, c2 = 2, 2
    w = w_s - ((w_s - w_e) * index/M_max)
    r1 = np.random.uniform()
    newVector = []
    for i in range(0, len(solution.params)):
        newParam = (vector[i] * w) + (r1 * c1 * (bestPSolution.params[i] - solution.params[i])) +  r1 * c2 * (bestGSolution.params[i] - solution.params[i])
        newVector.append(newParam)
    return newVector

def updatePosition(solution : Solution, vector):
    new_population = Solution(len(solution.params), solution.lower, solution.upper)
    for i in range(0, len(solution.params)):
        new_population.params[i] = solution.params[i] + vector[i]
    return new_population

def ParticleSwarm(solution, func):
    for i in range(0,len(solution.params)):
        solution.params[i] = random.uniform(solution.lower, solution.upper) # Generate random coordinates for the first time
    evaluationTracker = et.EvaluationTracker()
    pop_size = 30
    v_min, v_max = -1, 1
    population = generatePopulation(solution, pop_size) # Get population
    population = evaluateAll(population, func, evaluationTracker) # Set evaluations for population
    gBest = pickBest(population) # Best Global population
    vectors = generateVectors(population=population, v_min=v_min, v_max=v_max) # velocities for populations
    bestPopulation = list(np.copy(population)) # At first best versions of populations are the same as initial populations
    M_max = 50
    m = 0
    progressTracker = []
    while m < M_max :
        for i in range(0, len(population)):
            vectors[i] = updateVector(solution=population[i], bestPSolution = bestPopulation[i], bestGSolution=gBest, vector=vectors[i], index=i, M_max=M_max) # Update current velocity
            vectors[i] = np.clip(vectors[i], v_min, v_max) # Check boundaries
            population[i] = updatePosition(population[i], vectors[i]) # Update position (Generates new Solution)
            population[i].params = np.clip(population[i].params, population[i].lower, population[i].upper) # Check boundaries
            population[i].f = func(population[i].params) # Update population's evaluation
            evaluationTracker.currentEvalCount += 1
            if evaluationTracker.currentEvalCount > evaluationTracker.maxEval:
                return progressTracker
            #Compare a new position of a particle x to its pBest
            if population[i].f < bestPopulation[i].f:
                bestPopulation[i] = population[i]
                if population[i].f < gBest.f:
                    gBest = population[i]
                    progressTracker.append((population[i].params.copy()[0], population[i].params.copy()[1], population[i].f))
        m += 1
    return progressTracker