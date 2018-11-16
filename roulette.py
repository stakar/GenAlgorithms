#This is code for Roulette pooling

def roulette(population,target):
    generation = population_fitness(population,target)
    probability = 0
    wheel = {}
    if np.any(generation != 0):
        for individual in range(len(population)):
            if generation[individual] > 0:
                ind_probability = probability + (
                generation[individual] / np.sum(generation))
                wheel[individual] = ind_probability

                #Solution above fucks everything up, you must find new one

        for n in range(len(population)):
            print(np.random.random())
        return wheel #wheel.keys()
    else:
        return population

def fucking(parents):
    children = np.chararray((2,parents.shape[1]),unicode=True)
    n_heritage_low = np.random.randint(0,parents[0].shape[0])
    n_heritage_high = np.random.randint(n_heritage_low,
                                        parents[0].shape[0])
    children[0] = np.concatenate([parents[0][:n_heritage_low],
                                  parents[1][n_heritage_low:n_heritage_high],
                                  parents[0][n_heritage_high:]])

    children[0] = np.concatenate([parents[1][:n_heritage_low],
                                  parents[0][n_heritage_low:n_heritage_high],
                                  parents[1][n_heritage_high:]])
    return children


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

def roulette_swing(wheel):
    which = np.random.random()
    for n in range(len(wheel)):
        if which > wheel[n][1] and which < wheel[n][2]:
            return wheel[n][0]

def roulette(population,target,K):
    wheel = roulette_wheel(population,target)
    winners = np.chararray((K,population.shape[1]),unicode=True)
    for n in range(K):
        which = roulette_swing(wheel)
        winners[n] = population[which]
    return winners
