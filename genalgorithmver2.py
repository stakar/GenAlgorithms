#This file is containing a simple genetic algorithm prepared for MD seminary
#presentation

import numpy as np
from string import ascii_letters, punctuation

class GenAlgorithmString(object):

    def __init__(self,symbols = (ascii_letters + punctuation),n_population,
                 n_generation = 10):
        self = self
        self.symbols = symbols
        self.n_population = n_population
        self.n_generation = n_generation

    @staticmethod
    def get_symbol():
        return symbols[np.random.randint(len(symbols))]
    def generate_population(n_population,n_genotype):
        self.population = np.chararray((n_population,n_genotype),unicode=True)
    def fit(aim = aim):
        target = np.chararray((len(aim)),unicode=True)
        for chunk in range(len(target)):
            target[chunk] = aim[chunk]
        self.target = target
    def transform():
        new_population = population.copy()
        for generation in range(n_generation):
            new_population = descendants_generation(population,target)
        return new_population
    @staticmethod
    def _pooling(n_genotype=n_genotype):
        chromosome = np.chararray((n_genotype),unicode=True)
        for locus in range(n_genotype):
            chromosome[locus] = get_symbol()
        return chromosome
    @staticmethod
    def _mutate_population(population,n_population=n_population):
        for individual in range(n_population):
            population[individual] = pooling()
        return population
    @staticmethod
    def _check_fitness(chromosome,target):
        return np.count_nonzero(chromosome[chromosome == target])/len(chromosome)
    @staticmethod
    def _fucking(parents):
        children = np.chararray((2,parents.shape[1]),unicode=True)
        n_heritage = np.random.randint(0,parents[0].shape[0])
        children[0] = np.concatenate([parents[0][:n_heritage],
                                      parents[1][n_heritage:]])
        children[1] = np.concatenate([parents[1][:n_heritage],
                                      parents[0][n_heritage:]])
        return children

    @staticmethod
    def _population_fitness(population,target):
        fitness = np.zeros(population.shape[0])
        for individual in range(population.shape[0]):
            fitness[individual] = check_fitness(population[individual],target)
        return fitness

    @staticmethod
    def _ranking(population,target,K=5):
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
    @staticmethod
    def random_mutation(population):
        N,C = population.shape
        new_population = population.copy()
        n_mutations = round(population.size * 0.02)
        for n in range(n_mutations):
            new_population[np.random.randint(N),
                           np.random.randint(C)] = get_symbol()
        return new_population
    @staticmethod
    def descendants_generation(population,target,K=5,method = 'roulette'):
        if method == 'ranking':
            bests = ranking(population,target,K=K)
        elif method == 'roulette':
            bests = roulette(population,target,K=K)
        new_population = np.chararray((population.shape),unicode=True)
        for n in range(round(population.shape[0]/2)):
            parent1 = bests[np.random.randint(K)]
            parent2 = bests[np.random.randint(K)]
            descendants = fucking(np.array([parent1,parent2]))
            population[(n*2)] = descendants[0].squeeze()
            population[(n*2)+1] = descendants[1].squeeze()
            population = random_mutation(population)
        return population
    @staticmethod
    def roulette_wheel(population,target):
        generation = population_fitness(population,target)
        probability = 0
        wheel = np.zeros(3)
        if np.any(generation != 0):
            for individual in range(len(population)):
                if generation[individual] > 0:
                    ind_probability = probability + (
                    generation[individual] / np.sum(generation))
                    wheel = np.vstack([wheel,[individual,
                                       probability,ind_probability]])
                    probability = probability + (
                                  generation[individual] / np.sum(generation))
            return wheel[1:,:]
    @staticmethod
    def roulette_swing(wheel):
        which = np.random.random()
        for n in range(len(wheel)):
            if which > wheel[n][1] and which < wheel[n][2]:
                return int(wheel[n][0])
    @staticmethod
    def roulette(population,target,K):
        wheel = roulette_wheel(population,target)
        winners = np.chararray((K,population.shape[1]),unicode=True)
        for n in range(K):
            which = roulette_swing(wheel)
            winners[n] = population[which]
        return winners

#TODO create roulette selection of parents

# aim = "Programming is awesome!"
# n_genotype = len(aim)
# n_population = 10
