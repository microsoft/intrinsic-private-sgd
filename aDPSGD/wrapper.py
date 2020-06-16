import argparse
import numpy as np

from experiment_metadata import get_dataset_size
from run_experiment import load_cfg, run_experiment
from results_utils import ExperimentIdentifier
from derived_results import generate_derived_results


def run_sweep(cfg, num_seeds, num_replaces):
    print('Running sweep!')
    seeds = np.random.choice(99999, num_seeds)

    N = get_dataset_size(cfg['data'])
    replace_indices = np.random.choice(N, num_replaces)

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


def run_generate_derived_results(cfg, t):
    print('Generating derived results!')
    generate_derived_results(cfg['cfg_name'], model=cfg['model']['architecture'], t=t)


def run_produce_figures(cfg):
    print('Producing figures!')
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('switch', type=str, help='What part of workflow to run?',
                        choices=['sweep', 'derive', 'figures'])
    parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
    # --- these options are for switch == sweep
    parser.add_argument('--num_seeds', type=int, help='Number of seeds to run', default=10)
    parser.add_argument('--num_replaces', type=int, help='Number of replace indices to run', default=10)
    # --- these options are for switch == derive
    parser.add_argument('--t', type=int, help='Time point at which to run derived experiments', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    if args.switch == 'sweep':
        run_sweep(cfg, args.num_seeds, args.num_replaces)
    elif args.switch == 'derive':
        run_generate_derived_results(cfg, args.t)
    elif args.switch == 'figures':
        run_produce_figures(cfg)
