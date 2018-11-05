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
population = np.chararray((n_population,n_genotype),unicode=True)

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

def mutate_population(n_population=n_population):
    for individual in range(n_population):
        population[individual] = pooling()
    return population

def check_fitness(chromosome,target):
    return np.count_nonzero(chromosome[chromosome == target])/len(chromosome)

if __name__ == '__main__':
    pop = mutate_population()
    target = code_fit_func(aim)
    for n in range(pop.shape[0]):
        print(np.max(check_fitness(pop[n],target)))
    # print(check_fitness(pop[0],target


# notes:
# import numpy as np
# from string import ascii_letters,punctuation
# symbols = ascii_letters + punctuation
# def get_letter():
#     return symbols[np.random.randint(len(symbols))]
# string1 = np.chararray((22),unicode=True)
# string2 = np.chararray((22),unicode=True)
# for n in range(22):
#     string1[n] = get_letter()
# for n in range(22):
#     string2[n] = 'Programmin is awesome!'[n]
#
# string1 == string2
# %history
