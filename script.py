from genalgorithm import GenAlgorithmString as GA
from sys import argv

ga = GA(n_population=10,desired_fitness=0,mutation_probability=0.02)

#Preparing sentence. If sentence is given as argument in terminal, then it accep
#ts it and executes algorithm using it.
if len(argv) > 1:
    aim = argv[1]
else:
    aim = 'Code!'

print('Aim is: ' + aim)

# Fitting the sentence to the algorithm
ga.fit(aim)

#Printing first population
print('First population:')
print(ga.population)

#Printing first population's fitness
print('First population\'s fitness')
print(ga._population_fitness(ga.population))

#Executing an algorithm
ga.transform()

#Printing last population
print('Last population:')
print(ga.population)

#Printing best individual
print('best_individual:')
print(ga.best_individual(ga.population))

#Plotting the performance function
ga.plot_fitness()
