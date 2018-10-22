#This file is containing a simple genetic algorithm prepared for MD seminary
#presentation

import numpy as np
from string import ascii_letters, punctuation
symbols = ascii_letters + punctuation

def get_symbol():
    return symbols[np.random.randint(len(symbols))]

aim = "Programming is awesome!"
n_genotype = len(aim)


def pooling(n_genotype=n_genotype):
    chromosome = np.chararray((n_genotype),unicode=True)
    for genotype in range(n_genotype):
        chromosome[genotype] = get_symbol()
    return chromosome

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
