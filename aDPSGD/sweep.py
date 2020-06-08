import argparse
import numpy as np

from experiment_metadata import get_dataset_size
from run_experiment import load_cfg, run_experiment
from results_utils import ExperimentIdentifier

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
parser.add_argument('--num_seeds', type=int, help='Number of seeds to run', default=10)
parser.add_argument('--num_replaces', type=int, help='Number of replace indices to run', default=10)
args = parser.parse_args()
cfg = load_cfg(args.cfg)

seeds = np.random.choice(99999, args.num_seeds)

N = get_dataset_size(cfg['data'])
replace_indices = np.random.choice(N, args.num_replaces)

for seed in seeds:
    for replace_index in replace_indices:
        experiment = ExperimentIdentifier()
        experiment.init_from_cfg(cfg)
        for diffinit in False, True:
            experiment.diffinit = diffinit
            if experiment.exists():
                print(f'Experiment with seed {seed} and replace index {replace_index} already exists - skipping')
            else:
                run_experiment(cfg, diffinit, seed, replace_index)
