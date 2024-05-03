
import argparse



import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_pop", type=int, default=32)
parser.add_argument("--n_gen", type=int, default=100)
parser.add_argument("--n_parents", type=int, default=32)
parser.add_argument("--mr", type=int, default=None)


search_space = {
    "x": np.linspace(-10, 10, 1000),
    "y": np.linspace(-10, 10, 1000),
    "z": np.linspace(-10, 10, 1000),
    "x1": np.linspace(-10, 10, 1000),
    "y2": np.linspace(-10, 10, 1000),
    "z3": np.linspace(-10, 10, 1000),
    "x5": np.linspace(-10, 10, 1000),
    "y6": np.linspace(-10, 10, 1000),
    "z7": np.linspace(-10, 10, 1000),
    "a5": np.linspace(-10, 10, 1000),
    "b6": np.linspace(-10, 10, 1000),
    "c7": np.linspace(-10, 10, 1000),
    "1x": np.linspace(-10, 10, 1000),
    "1y": np.linspace(-10, 10, 1000),
    "1z": np.linspace(-10, 10, 1000),
    "1x1": np.linspace(-10, 10, 1000),
    "1y2": np.linspace(-10, 10, 1000),
    "1z3": np.linspace(-10, 10, 1000),
    "1x5": np.linspace(-10, 10, 1000),
    "1y6": np.linspace(-10, 10, 1000),
    "1z7": np.linspace(-10, 10, 1000),
    "1a5": np.linspace(-10, 10, 1000),
    "1b6": np.linspace(-10, 10, 1000),
    "1c7": np.linspace(-10, 10, 1000),
}


def evaluate_function(x):
    return np.sum([-x[k]**2 for k in search_space])

def genetic_algorithm():
    np.random.seed(0)

    ngen = 100
    n_parents = 16
    npop = 32
    mr = 0.01
    if isinstance(mr, float):
        mr = {k: mr for k in search_space.keys()}

    population = [{k: np.random.choice(v) for k, v in search_space.items()} for _ in range(npop)]
    fitness = [evaluate_function(x) for x in population]

    def mutate_fn(x, mr):
        does_mutate = {k: np.random.rand() < mri for k, mri in mr.items()}
        x_new = {k: (np.random.choice(search_space[k]) if does_mutate[k] else x[k]) for k in search_space}
        return x_new

    a = []
    for i in range(ngen):
        idx_best = np.argmax(fitness)
        idx_sort = np.argsort(fitness)[::-1]

        elite = population[idx_best]
        selected = [population[idx] for idx in idx_sort[:n_parents]]

        parents = [np.random.choice(selected) for _ in range(npop-1)]
        children = [mutate_fn(parent, mr) for parent in parents]

        population = [elite] + children
        fitness = [evaluate_function(x) for x in population]
        a.append(fitness)
    a = np.array(a)
    plt.plot(a.mean(axis=1))
    plt.plot(a.max(axis=1))
    plt.plot(a.min(axis=1))
    plt.show()


if __name__=="__main__":
    genetic_algorithm()
    # import yaml

    # with open("arch.yaml", "r") as f:
    #     a = yaml.safe_load(f)
    
    # print(a)








