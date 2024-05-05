from typing import Optional
import os
import threading
import timeloopfe.v4 as tl
import argparse
import pickle
import json
import numpy as np
import sys
from tqdm.auto import tqdm
import pandas as pd

EXAMPLE_DIR = "example_designs"
TOP_JINJA_PATH = "example_designs/top.yaml.jinja2"
###############################################################################                                   
# Command line arguments
######################## 
def getArgumentParser():
    """ Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Script to run iteration of timeloop and get output stats.")
    parser.add_argument('-o',
                        '--output_file',
                        dest = 'output_file',
                        help='output file',
                        default = "output.pkl")
    #### ADD OTHER ARGUMENTS TO THE .YAML FILE HERE ####
    return parser


def run_mapper(arch_target, problem, output_file, design_params):
    jinja_parse_data = {'architecture': arch_target}

    ### SEE RUN_EXAMPLE.PY, NEED TO PUT TOGETHER A LIST OF PROBLEMS TO RUN!!!
    jinja_parse_data["problem"] = problem
    # output dir to get the stats from 
    output_dir = "output"
    if os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    spec = tl.Specification.from_yaml_files(TOP_JINJA_PATH,jinja_parse_data = jinja_parse_data)

    print(spec.architecture.find("shared_glb").attributes)
    print(spec.architecture.find("PE_column").spatial)

    for k, v in design_params.items():
        k1, k2, k3 = k.split("/")
        buf = spec.architecture.find(k1)
        getattr(buf, k2)[k3] = v

    print(spec.architecture.find("shared_glb").attributes)
    print(spec.architecture.find("PE_column").spatial)


    tl.call_mapper(
        spec,
        output_dir="output",
        #dump_intermediate_to=output_dir,
    )

    stats = open("output/timeloop-mapper.stats.txt").read()
    stats = [l.strip() for l in stats.split("\n") if l.strip()]
    # Find the index where 'Summary Stats' appears
    summary_stats_index = stats.index('Summary Stats')
    
    # Extract the summary stats
    summary_stats = stats[summary_stats_index + 2:summary_stats_index + 8]  # Assuming summary stats take 8 lines
    stat_dict = {}
    for stat in summary_stats:
        key, val = stat.split(":")
        key = key.strip()
        val = val.strip()
        if key == "Utilization":
            key = "Utilization %"
            val = float(val.split('%')[0])
        if key == "Energy":
            key = "Energy (uJ)"
            val = float(val.split(' ')[0])
        if key == "Area":
            key = "Area (mm^2)"
            val = float(val.split(' ')[0])
        else:
            val = float(val)
        stat_dict[key] = val
    energy = float(stats[-1].split("=")[-1])
    stat_dict['Total Energy (fJ/Compute)'] = energy
    print(stat_dict)
    # with open(output_file, 'wb') as f:
        # pickle.dump(stat_dict, f)
    return stat_dict

# def main():
#     arch_target = 'eyeriss_like'
#     problem = os.path.join("layer_shapes","vgg16")
#     # all possible problems here
#     problems = [os.path.join("..",problem, f) for f in os.listdir(problem)]
#     options = getArgumentParser().parse_args()
#     outfile = options.output_file

#     design_params = {
#         # "shared_glb/attributes/depth": 1000,
#         # "PE_column/spatial/meshX": 12,
#     }
#     run_mapper(arch_target, problems[0], outfile, design_params=design_params)

def evaluate_function(x):
    arch_target = 'eyeriss_like'
    problem = os.path.join("layer_shapes","vgg16")
    problems = [os.path.join("..",problem, f) for f in os.listdir(problem)]
    # options = getArgumentParser().parse_args()
    # outfile = options.output_file

    design_params = {k: int(v) for k, v in x.items()}
    design_params["PE/spatial/meshY"] = 168//design_params["PE_column/spatial/meshX"]

    metrics = {}

    try:
        sys.stdout = open('/dev/null', 'w')
        metrics = run_mapper(arch_target, problems[0], None, design_params=design_params)
        sys.stdout = sys.__stdout__
        metrics['fitness'] = -metrics["Total Energy (fJ/Compute)"]
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(e)
        metrics['fitness'] = -float('inf')
    return metrics

# def evaluate_function(x):
    # return -sum([x[k] for k in x])


search_space = {
    "shared_glb/attributes/depth": [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    "shared_glb/attributes/width": [16, 32, 64, 128, 256, 512],
    "PE_column/spatial/meshX": [1, 2, 3, 4, 6, 7, 8, 12, 14, 21, 24, 28, 42, 56, 84, 168],
    "PE/spatial/meshY": [1, 2, 3, 4, 6, 7, 8, 12, 14, 21, 24, 28, 42, 56, 84, 168],
    "ifmap_spad/attributes/depth": [3, 6, 12, 24, 48],
    "ifmap_spad/attributes/width": [8, 16, 32, 64],

    "weights_spad/attributes/depth": [48, 96, 192, 384,],
    "weights_spad/attributes/width": [8, 16, 32, 64],

    "psum_spad/attributes/depth": [8, 16, 32, 64],
    "psum_spad/attributes/width": [16, 32, 64, 128],

}

def main_random():
    np.random.seed(0)
    metrics = []
    for i in tqdm(range(100)):
        design_params = {k: int(np.random.choice(v)) for k, v in search_space.items()}
        metrics.append(evaluate_function(design_params))
        print(metrics[-1])
    with open("metrics_random.json", "w") as f:
        json.dump(metrics, f)

def main_genetic_algo():
    with open("metrics_random.json", "r") as f:
        metrics_random = json.load(f)
    metrics_random = {k: [dic[k] for dic in metrics_random] for k in metrics_random[0].keys()}
    df = pd.DataFrame(metrics_random)

    coef_gflops = 1.
    coef_util = 1.
    coef_cycles = -1.
    coef_energy = -1.

    def my_evaluate_fn(x):
        metrics = evaluate_function(x)

        if metrics['fitness'] == -float('inf'):
            pass
        else:
            gflops = metrics['GFLOPs (@1GHz)']
            util = metrics['Utilization %']
            cycles = metrics['Cycles']
            energy = metrics['Total Energy (fJ/Compute)']

            gflops = (gflops - df['GFLOPs (@1GHz)'].mean()) / df['GFLOPs (@1GHz)'].std()
            util = (util - df['Utilization %'].mean()) / df['Utilization %'].std()
            cycles = (cycles - df['Cycles'].mean()) / df['Cycles'].std()
            energy = (energy - df['Total Energy (fJ/Compute)'].mean()) / df['Total Energy (fJ/Compute)'].std()

            fitness = coef_gflops * gflops + coef_util * util + coef_cycles * cycles + coef_energy * energy

            metrics['fitness'] = fitness
        return metrics

    genetic_algorithm(0, ngen=500, npop=8, n_parents=6, mr=0.1, search_space=search_space, evaluate_function=my_evaluate_fn)
    # genetic_algorithm(0, ngen=100, npop=16, n_parents=8, mr=0.1, search_space=search_space, evaluate_function=my_evaluate_fn)


def genetic_algorithm(seed, ngen=100, npop=32, n_parents=16, mr=0.01, search_space=None, evaluate_function=None):
    np.random.seed(seed)
    if isinstance(mr, float):
        mr = {k: mr for k in search_space.keys()}

    def eval_pop_sequential(population):
        metrics = [evaluate_function(x) for x in population]
        fitnesses = [m['fitness'] for m in metrics]
        return fitnesses, metrics

    # def eval_pop_parallel(population):
    #     from multiprocessing import Pool
    #     with Pool() as pool:
    #         # fitnesses = pool.map(evaluate_function, population)
    #         fitnesses = pool.map(evaluate_function, population)
    #     return fitnesses
    
    eval_pop_fn = eval_pop_sequential

    population_history = []
    metrics_history = []

    population = [{k: np.random.choice(v) for k, v in search_space.items()} for _ in range(npop)]
    fitness, metrics = eval_pop_fn(population)


    def mutate_fn(x, mr):
        does_mutate = {k: np.random.rand() < mri for k, mri in mr.items()}
        x_new = {k: (np.random.choice(search_space[k]) if does_mutate[k] else x[k]) for k in search_space}
        return x_new

    for i in tqdm(range(ngen)):
        population_history.append(population)
        metrics_history.append(metrics)
        print(f"Mean fitness: {np.mean(fitness):.2f}, Max fitness: {np.max(fitness):.2f}, Min fitness: {np.min(fitness):.2f}")

        idx_best = np.argmax(fitness)
        idx_sort = np.argsort(fitness)[::-1]

        elite = population[idx_best]
        selected = [population[idx] for idx in idx_sort[:n_parents]]

        parents = [np.random.choice(selected) for _ in range(npop-1)]
        children = [mutate_fn(parent, mr) for parent in parents]

        population = [elite] + children
        fitness, metrics = eval_pop_fn(population)

        if i%5 == 0:
            with open("ga_history.pkl", "wb") as f:
                pickle.dump(dict(population_history=population_history, metrics_history=metrics_history), f)

    population_history.append(population)
    metrics_history.append(metrics)
    with open("ga_history.pkl", "wb") as f:
        pickle.dump(dict(population_history=population_history, metrics_history=metrics_history), f)
        


if __name__ == "__main__":
    main_genetic_algo()



