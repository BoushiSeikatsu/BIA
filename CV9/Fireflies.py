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

def calculateDistance(firstFirefly, secondFirefly):
    distance = 0
    for i in range(len(firstFirefly.params)):
        distance += pow(firstFirefly.params[i] - secondFirefly.params[i], 2)
    return math.sqrt(distance)

def evaluateAll(population : list[Solution], func, evaluationTracker : et.EvaluationTracker):
    for firefly in population:
        firefly.f = func(firefly.params)
        evaluationTracker.currentEvalCount += 1

# MIght not get used and instead we will just use func evaluations instead of light intensities 
def lightIntensities(population : list[Solution], distanceMatrix, gamma = 0.5):
    lightIntens = []
    for _ in range(len(population)):
        lightIntens.append([])
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                intensity0 = population[i].f # We calculate light intensity for i firefly 
                distance = calculateDistance(population[i], population[j]) # Calculate the distance between i firefly and j firefly
                intensity = intensity0 * pow(math.e,-gamma*distanceMatrix[i][j])
                lightIntens[i].append(intensity)

def pickBest(population : Solution):
    bestScore = sys.maxsize
    bestSolution = None
    for solution in population:
        if solution.f < bestScore:
            bestScore = solution.f
            bestSolution = solution
    return copy.copy(bestSolution)

def Fireflies(solution, func):
    evaluationTracker = et.EvaluationTracker()
    for i in range(0,len(solution.params)):
        solution.params[i] = random.uniform(solution.lower, solution.upper) # Generate random coordinates for the first time
    #Initialize a population of fireflies
    n_population = 30
    population = generatePopulation(solution, n_population)
    evaluateAll(population, func, evaluationTracker=evaluationTracker)
    #Set control parameters
    B_0 = 1
    gamma = 0.5
    alpha = 0.3
    max_iter = 200
    bestSolution = pickBest(population)
    progressTracker = []
    #t = 0
    for _ in range(0, max_iter):
        for i in range(n_population):
            for j in range(n_population):
                if population[i].f > population[j].f: # Equivalent to I_j > I_i
                    r = calculateDistance(population[i], population[j]) # Calculate distance r
                    beta = B_0/(1 + r) # Calculate beta 
                    randomVal = np.random.normal(0, 1, solution.d) # Random value e from normal distribution
                    for k in range(len(population[i].params)): # For each param
                        population[i].params[k] += beta * (population[j].params[k] - population[i].params[k]) + alpha * randomVal[k] # Formula for fireflies coming closer 
                    population[i].params = np.clip(population[i].params, solution.lower, solution.upper) # Ensure we are not out of bounds
                    population[i].f = func(population[i].params)
                    evaluationTracker.currentEvalCount += 1
                    if evaluationTracker.currentEvalCount > evaluationTracker.maxEval: # If max eval count has been reached, finish algorithm
                        return progressTracker                   
                    #Move firefly i towards j
        potentialBest = pickBest(population)
        if potentialBest.f < bestSolution.f:
            bestSolution = potentialBest
            progressTracker.append((bestSolution.params.copy()[0], bestSolution.params.copy()[1], bestSolution.f))
        #Rank the fireflies and find the current best firefly
    return progressTracker