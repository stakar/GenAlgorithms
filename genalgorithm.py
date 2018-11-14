#This file is containing a simple genetic algorithm prepared for MD seminary
#presentation

import numpy as np
from string import ascii_letters, punctuation
symbols = ascii_letters + punctuation

def get_symbol():
    return symbols[np.random.randint(len(symbols))]

aim = "Programming is awesome!"
n_genotype = len(aim)
n_population = 5

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

def fucking(parents):
    children = np.chararray((2,parents.shape[1]),unicode=True)
    n_heritage = np.random.randint(5,parents[0].shape[0])
    children[0] = np.concatenate([parents[0][:n_heritage],parents[1][n_heritage:]])
    children[1] = np.concatenate([parents[1][:n_heritage],parents[0][n_heritage:]])
    return children

def population_fitness(population,target):
    fitness = np.zeros(population.shape[0])
    for individual in range(population.shape[0]):
        fitness[individual] = check_fitness(population[individual],target)
    return fitness

def tournament(population,target,K=5):
    fitness = np.zeros(population.shape[0])
    for individual in range(population.shape[0]):
        fitness[individual] = check_fitness(population[individual],target)
    population_of_best = np.chararray((K,population.shape[1]),unicode=True)
    if np.any(fitness != 0):
        for k in range(K):
            population_of_best[k] = population[np.where(np.max(fitness))]
            population = np.delete(population,np.where(np.max(fitness)),0)
        return population_of_best
    else:
        return mutate_population(population)

def roulette():
    pass

def random_mutation(population):
    N,C = population.shape
    new_population = population.copy()
    n_mutations = round(population.size * 0.02)
    for n in range(n_mutations):
        new_population[np.random.randint(N),np.random.randint(C)] = get_symbol()
    return new_population

#Be aware that in some populations every individual could have 0 fitness

def descendants_generation(population,target,K=5,method = 'tournament'):
    if method == 'tournament':
        bests = tournament(population,target,K=K)
    new_population = np.chararray((population.shape),unicode=True)
    for n in range(round(population.shape[0]/2)):
        parent1 = bests[np.random.randint(K)]
        parent2 = bests[np.random.randint(K)]
        descendants = fucking(np.array([parent1,parent2]))
        population[(n*2)] = descendants[0].squeeze()
        population[(n*2)+1] = descendants[1].squeeze()
        population = random_mutation(population)
    return population

#TODO create roulette selection of parents

if __name__ == '__main__':
    population = generate_population(n_population,n_genotype)
    pop = mutate_population(population)
    target = code_fit_func(aim
    # print(population_fitness(pop,target))
    new = descendants_generation(pop,target)
    for n in range(100):
        new = descendants_generation(pop,target)
    print(population_fitness(new,target))
    print(new[np.where(np.max(population_fitness(new,target)))])
