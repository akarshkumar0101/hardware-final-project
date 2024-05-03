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


def run_mapper(arch_target, problem, output_file):
    jinja_parse_data = {'architecture': arch_target}

    ### SEE RUN_EXAMPLE.PY, NEED TO PUT TOGETHER A LIST OF PROBLEMS TO RUN!!!
    jinja_parse_data["problem"] = problem
    # output dir to get the stats from 
    output_dir = "output"
    if os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    spec = tl.Specification.from_yaml_files(TOP_JINJA_PATH,jinja_parse_data = jinja_parse_data)
    #print(spec)
    #buf = spec.architecture.find("shared_glb")
    #buf.attributes["depth"] = 1000
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
    with open(output_file, 'wb') as f:
        pickle.dump(stat_dict, f)

def main():
    arch_target = 'eyeriss_like'
    problem = os.path.join("layer_shapes","vgg16")
    # all possible problems here
    problems = [os.path.join("..",problem, f) for f in os.listdir(problem)]
    print(problems)
    options = getArgumentParser().parse_args()
    outfile = options.output_file
    run_mapper(arch_target, problems[13], outfile)

if __name__ == "__main__":
    main()
