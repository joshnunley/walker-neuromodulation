import matplotlib.pyplot as plt
import numpy as np

from CTRNN import CTRNN
from EvolSearch import EvolSearch
from fitness_function_initial import fitness_function_initial

import pickle

# WARNING I AM FILTERING WARNINGS BECAUSE PATHOS DOESN'T LIKE THEM
import warnings

warnings.filterwarnings("ignore")

use_best_individual = False
if use_best_individual:
    with open("best_individual", "rb") as f:
       best_individual = pickle.load(f)

########################
# Parameters
########################
ctrnn_size = 100

num_params = int(ctrnn_size ** 2 + 2 * ctrnn_size)
ctrnn_params = np.random.rand(num_params)

settings = {
   "ctrnn_size": ctrnn_size,
   "ctrnn_step_size": 0.05,
   "ctrnn_params": ctrnn_params,
   "walker_duration": 150,
   "walker_step_size": 0.05,
   "transient_steps": 100,
}

########################
# Evolve Solutions
########################

pop_size = 5
genotype_size = ctrnn_size


evol_params = {
    "num_processes": 10,
    "pop_size": pop_size,  # population size
    "genotype_size": genotype_size,  # dimensionality of solution
    "fitness_function": lambda initial_outputs: fitness_function_initial(initial_outputs, **settings),  # custom function defined to evaluate fitness of a solution
    "elitist_fraction": 0.1,  # fraction of population retained as is between generation
    "mutation_variance": 0.05,  # mutation noise added to offspring.
}
#initial_pop = np.random.choice([0.0, 0.5, 1.0], p=[0.34, 0.33, 0.33], size=(pop_size, genotype_size))
initial_pop = np.random.uniform(size=(pop_size, genotype_size))
if use_best_individual:
    initial_pop[0] = best_individual["params"]

evolution = EvolSearch(evol_params, initial_pop)

save_best_individual = {
   "params": None,
   "best_fitness": [],
   "mean_fitness": [],
   "settings": settings,
}

for i in range(40):
    evolution.step_generation()
    
    save_best_individual["params"] = evolution.get_best_individual()
    
    save_best_individual["best_fitness"].append(evolution.get_best_individual_fitness())
    save_best_individual["mean_fitness"].append(evolution.get_mean_fitness())

    print(
        len(save_best_individual["best_fitness"]), 
        save_best_individual["best_fitness"][-1], 
        save_best_individual["mean_fitness"][-1]
    )

    with open("best_individual", "wb") as f:
        pickle.dump(save_best_individual, f)

