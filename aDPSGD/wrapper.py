import argparse
from run_experiment import run_single_experiment, find_gaps_in_grid, add_new_seeds_to_grid, propose_seeds_and_replaces
from cfg_utils import load_cfg
from results_utils import ExperimentIdentifier
from derived_results import generate_derived_results
from produce_figures import generate_plots, generate_reports


def run_sweep(cfg, num_seeds, num_replaces, fill_in_gaps: bool = False, add_seeds: bool = False):
    print('Running sweep!')
    if fill_in_gaps:
        print('Flag fill_in_gaps provided as True - not running any *new* seeds or replaces!')
        pairs = find_gaps_in_grid(cfg)
    elif add_seeds:
        print('Adding seeds!')
        pairs = add_new_seeds_to_grid(cfg, num_seeds, num_replaces)
    else:
        print(f'Running {num_seeds} new seeds and {num_replaces} new replaces!')
        pairs = propose_seeds_and_replaces(cfg, num_seeds, num_replaces)

    print(f'Note! We are about to run {len(pairs)} experiments.')
    for seed, replace_index in pairs:
        experiment = ExperimentIdentifier()
        experiment.init_from_cfg(cfg)
        experiment.seed = seed
        experiment.replace_index = replace_index

        for diffinit in False, True:
            experiment.diffinit = diffinit

            if experiment.exists():
                print(f'\t\t[sweep] Experiment with seed {seed} and replace index {replace_index} already exists - skipping')
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
    parser.add_argument('--fill_in_gaps', type=bool, help='Whether to fill in gaps in the grid', default=False)
    parser.add_argument('--add_seeds', type=bool, help='Whether we are simply adding seeds', default=False)
    # --- these options are for switch == derive
    parser.add_argument('--t', type=int, help='Time point at which to run derived experiments', default=None)
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    if args.switch == 'sweep':
        run_sweep(cfg, args.num_seeds, args.num_replaces, args.fill_in_gaps, args.add_seeds)
    elif args.switch == 'derive':
        run_generate_derived_results(cfg, args.t)
    elif args.switch == 'figures':
        run_produce_figures(cfg, args.t)
    elif args.switch == 'report':
        run_generate_reports(cfg, args.t)
