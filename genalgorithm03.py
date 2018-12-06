import numpy as np
from string import ascii_letters, punctuation

class GenAlgorithmString(object):

    def __init__(self,symbols = (ascii_letters + punctuation + ' '),
                 n_population=10,method = 'roulette',K=4,desired_fitness=0.4,
                 mutation_probability = 0.02):

        self = self
        self.symbols = symbols
        self.n_population = n_population
        self.method = method
        self.K = K
        self.desired_fitness = desired_fitness
        self.mutation_probability = mutation_probability

    def get_symbol(self):
        """ Generates random symbol """
        return self.symbols[np.random.randint(len(self.symbols))]

    def fit(self,aim):
        """ Fits the given sentence to the mod el. Input aim is the string, sente
        nce that the algorithm is supposed to found."""
        self.target = np.array([n for n in aim],dtype = 'U1')
        self.n_genotype = len(self.target)
        self.population = np.array([[self.get_symbol() for
         n in range(self.n_genotype)] for n in range(self.n_population)])

    def transform():
        """ Transform, i.e. execute an algorithm. """
        best_fitness = np.min(self._population_fitness())
        self.n_generation = 0
        while best_fitness < self.desired_fitness:
            self(self.descendants_generation())
            self.n_generation += 1
            best_fitness = np.min(self._population_fitness())


    def _check_fitness(self,chromosome,target):
        """ Checks the fitness of individual. Fitness is the mesure of distance
        between letter generated randomly and the one that maps it on the locus
        in target array"""
        return sum([abs(ord(chromosome[n])-ord(target[n])) for n in range(
                       self.n_genotype)])

    def _population_fitness(self):
        """Returns an arrray with fitness of each individual in population"""
        return np.array([self._check_fitness(n,self.target) for
                         n in self.population])

    @staticmethod
    def _pairing(mother,father):
        """ Method for pairing chromosomes and generating descendants, array of
        characters with shape [2,n_genotype] """
        n_heritage = np.random.randint(0,len(mother))
        child1 = np.concatenate([father[:n_heritage],mother[n_heritage:]])
        child2 = np.concatenate([mother[:n_heritage],father[n_heritage:]])
        return child1,child2

    def descendants_generation(self):
        """ Selects the best individuals, then generates new population, with
        half made of parents (i.e. best individuals) and half children(descendan
        ts of parents) """
        parents_pop = self.roulette()
        n_parents = int(np.round(self.n_population/3))
        for n in range(n_parents,self.n_population-1):
            father = parents_pop[np.random.randint(self.n_population)]
            mother = parents_pop[np.random.randint(self.n_population)]
            children = self._pairing(mother,father)
            self.population[(n)] = children[0]
            self.population[(n)+1] = children[1]
        self.population[:n_parents] = parents_pop[:n_parents]

    def roulette_wheel(self):
        """ Method that returns roulette wheel, an array with shape [n_populatio
        n, low_individual_probability,high_individual_probability]"""
        max_val = 126*self.n_genotype
        pop_fitness = [max_val-n for n in self._population_fitness()]
        wheel = np.zeros((self.n_population,3))
        prob = 0
        for n in range(self.n_population):
            ind_prob = prob + (pop_fitness[n] / np.sum(pop_fitness))
            wheel[n] = [n,prob,ind_prob]
            prob = ind_prob
        return wheel

    def roulette_swing(self,wheel):
        """ This method takes as an inpute """
        which = np.random.random()
        for n in range(len(wheel)):
            if which > wheel[n][1] and which < wheel[n][2]:
                return int(wheel[n][0])

    def roulette(self):
        """ This method performs selection of individuals, it takes the coeffici
        ent k, which is number of new individuals """
        wheel = self.roulette_wheel()
        return np.array([self.population[self.roulette_swing(wheel)]
                         for n in range(self.n_population)])

    def random_mutation(self):
        population = self.population.copy()
        for n in range(self.n_population):
            decision = np.random.random()
            if decision < self.mutation_probability:
                which_gene = np.random.randint(self.n_genotype)
                population[n][which_gene] = self.get_symbol()
        self.population = population
if __name__ == '__main__':
    genalg = GenAlgorithmString(K=5,n_population=5,method = 'roulette',
                                desired_fitness = 50,mutation_probability=0.02)
    genalg.fit('Programming is awesome')
    pop1 = genalg.population
    print(pop1)
    genalg.random_mutation()
    # genalg.descendants_generation()
    pop2 = genalg.population
    print('CHildren:')
    print(pop2)
    print(pop1 == pop2)
