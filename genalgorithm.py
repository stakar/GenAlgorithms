#This file is containing a simple genetic algorithm prepared for MD seminary
#presentation

import numpy as np
from string import ascii_letters, punctuation
symbols = ascii_letters + punctuation

def get_symbol():
    return symbols[np.random.randint(len(symbols))]

aim = "Programming is awesome!"
n_genotype = len(aim)
n_population = 10

def generate_population(n_population,n_genotype):
    return np.chararray((n_population,n_genotype),unicode=True)

def code_fit_func(aim = aim):
    target = np.chararray((len(aim)),unicode=True)
    for chunk in range(len(target)):
        target[chunk] = aim[chunk]
    return target

def pooling(n_genotype=n_genotype):
    chromosome = np.chararray((n_genotype),unicode=True)
    for locus in range(n_genotype):
        chromosome[locus] = get_symbol()
    return chromosome

def mutate_population(population,n_population=n_population):
    for individual in range(n_population):
        population[individual] = pooling()
    return population

def check_fitness(chromosome,target):
    return np.count_nonzero(chromosome[chromosome == target])/len(chromosome)

def find_best(population,target,K=2):
    fitness = np.zeros(population.shape[0])
    for individual in range(population.shape[0]):
        fitness[individual] = check_fitness(population[individual],target)
    sorted = -np.sort(-fitness)[:K]
    # K_best = population[np.where(fitness[K:])]
    population_of_best = np.chararray((K,population.shape[1]),unicode=True)
    for k in range(K):
        population_of_best = population[np.where(fitness == sorted[k])]
    return population_of_best

#TODO work above function out



if __name__ == '__main__':
    population = generate_population(n_population,n_genotype)
    pop = mutate_population(population)
    target = code_fit_func(aim)
    result = np.zeros(pop.shape[0])
    # print(-np.sort(-result))
    print(find_best(population,target))
    # print(check_fitness(pop[0],target
