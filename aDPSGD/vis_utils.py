#!/usr/bin/env ipython
# visualisation tools!


def beautify_axes(axarr):
    """
    Standard prettification edits I do in matplotlib
    """

    if len(axarr.shape) == 1:
        axarr = [axarr]

    for axrow in axarr:
        for ax in axrow:
            ax.set_facecolor((0.95, 0.95, 0.95))
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(bottom=False, left=False)
            ax.grid(linestyle='--', alpha=0.5)

    return True


def process_identifiers(datasets, models, replaces, seeds, privacys):
    # figure out length of longest provided list
    n_identifiers = 1

    for identifier_component in datasets, replaces, seeds, privacys:
        if type(identifier_component) == list:
            length = len(identifier_component)

            if length > n_identifiers:
                n_identifiers = length
    # make sure everything is either that length or not a list

    if type(datasets) == list:
        if not len(datasets) == n_identifiers:
            assert len(datasets) == 1
            datasets = n_identifiers*datasets
    else:
        datasets = [datasets]*n_identifiers

    if type(models) == list:
        if not len(models) == n_identifiers:
            assert len(models) == 1
            models = n_identifiers*models
    else:
        models = [models]*n_identifiers

    if type(replaces) == list:
        if not len(replaces) == n_identifiers:
            assert len(replaces) == 1
            replaces = n_identifiers*replaces
    else:
        replaces = [replaces]*n_identifiers

    if type(seeds) == list:
        if not len(seeds) == n_identifiers:
            assert len(seeds) == 1
            seeds = n_identifiers*seeds
    else:
        seeds = [seeds]*n_identifiers

    if type(privacys) == list:
        if not len(privacys) == n_identifiers:
            assert len(privacys) == 1
            privacys = n_identifiers*privacys
    else:
        privacys = [privacys]*n_identifiers

    identifiers = []
    for i in range(n_identifiers):
        identifiers.append({'dataset': datasets[i],
                            'model': models[i],
                            'replace': replaces[i],
                            'seed': seeds[i],
                            'data_privacy': privacys[i]})

    return identifiers
