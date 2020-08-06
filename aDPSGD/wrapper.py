import argparse
from run_experiment import load_cfg, run_single_experiment
from results_utils import ExperimentIdentifier, get_grid_to_run
from derived_results import generate_derived_results
from produce_figures import generate_plots, generate_reports
import numpy as np


def run_sweep(cfg, num_seeds, num_replaces):
    print('Running sweep!')
    seeds, replace_indices = get_grid_to_run(cfg, num_seeds, num_replaces)
    seeds = np.random.choice(seeds, num_seeds, replace=False)
    replace_indices = np.random.choice(replace_indices, num_replaces, replace=False)

    for seed in seeds:
        for replace_index in replace_indices:
            experiment = ExperimentIdentifier()
            experiment.init_from_cfg(cfg)

            for diffinit in False, True:
                experiment.diffinit = diffinit

                if experiment.exists():
                    print(f'Experiment with seed {seed} and replace index {replace_index} already exists - skipping')
                else:
                    run_single_experiment(cfg, diffinit, seed, replace_index)


def run_generate_derived_results(cfg, t):
    print('Generating derived results!')
    generate_derived_results(cfg['cfg_name'], model=cfg['model']['architecture'], t=t)


def run_produce_figures(cfg, t):
    print('Producing figures!')
    generate_plots(cfg['cfg_name'], model=cfg['model']['architecture'], t=t)


def run_generate_reports(cfg, t):
    print('Generating report...')
    generate_reports(cfg['cfg_name'], model=cfg['model']['architecture'], t=t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('switch', type=str, help='What part of workflow to run?',
                        choices=['sweep', 'derive', 'figures', 'report'])
    parser.add_argument('--cfg', type=str, help='Name of yaml cfg of experiment')
    # --- these options are for switch == sweep
    parser.add_argument('--num_seeds', type=int, help='Number of seeds to run', default=100)
    parser.add_argument('--num_replaces', type=int, help='Number of replace indices to run', default=100)
    # --- these options are for switch == derive
    parser.add_argument('--t', type=int, help='Time point at which to run derived experiments', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    if args.switch == 'sweep':
        run_sweep(cfg, args.num_seeds, args.num_replaces)
    elif args.switch == 'derive':
        run_generate_derived_results(cfg, args.t)
    elif args.switch == 'figures':
        run_produce_figures(cfg, args.t)
    elif args.switch == 'report':
        run_generate_reports(cfg, args.t)
