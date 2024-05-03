from typing import Optional
import os
import threading
import timeloopfe.v4 as tl
import argparse
import pickle

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


def main():
    arch_target = 'eyeriss_like'
    problem = os.path.join("layer_shapes","vgg16")
    # all possible problems here
    problems = [os.path.join("..",problem, f) for f in os.listdir(problem)]
    options = getArgumentParser().parse_args()
    outfile = options.output_file



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

    # design_params = {
        # "shared_glb/attributes/depth": 1000,
        # "PE_column/spatial/meshX": 12,
    # }

    import numpy as np
    design_params = {k: int(np.random.choice(v)) for k, v in search_space.items()}

    run_mapper(arch_target, problems[0], outfile, design_params=design_params)



if __name__ == "__main__":
    main()



