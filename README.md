# Introduction 
The objective of this codebase is to allow empirical privacy analysis of SGD.

At the heart of this is running a large grid of short experiments and then processing and analysing the results.

# Directory structure

Expected subfolders:
```
aDPSGD
|-- cfgs
|-- data
|-- figures
|-- models
|-- traces
|-- visualisations
```

- `cfgs` contains experiment configs (explained below).
- `data` contains training/test data
- `figures` contains figures for e.g. a paper
- `models` contains the _fixed initialisations_ for each model class on each dataset
- `traces` contains the results from the experiments. This is recommended to be a symbolic link and its contents not committed to the repository, as the folder is expected to become very large (e.g. hundreds of GB). If the path to traces needs to be overriden, change the definition of `TRACES_DIR` in `results_utils.py`
- `visualisations` contains visualisations that are more throw-away (not for a paper)

The structure of `traces` follows the format
```
traces / cfg_identifier / dataset_privacy / model_class
```
- `cfg_identifier ` identifies the experiment, usually given by the data domain/task (e.g. `mnist_binary`)
- `dataset_privacy` is slightly vestigial, but usually `all`
- `model_class` is something like `logistic`

The contents of this folder will then be many thousand results files from models trained on variants of that dataset with different random seeds.
For example, if `traces` is a symbolic link to `/bigdata/traces_aDPSGD`, the path to the saved weights of a logistic regression model trained on the MNIST binary task might look like:
```
/bigdata/traces_aDPSGD/mnist_binary/all/logistic/logistic_DIFFINIT.replace_10270.seed_13774.weights.csv
```
In the above example, we used variable intialisation (`DIFFINIT`), replaced training example with index `10270` with `x_0`, and used seed `13774`.

# Workflow

The typical workflow follows these steps:
1. **Run a "sweep"** of experiments - this would be a grid of random seeds and "replace indices" for a single dataset for a single model class.
2. Aggregate across the results to **generate "derived" results**. These include estimates of the empirical sensitivity and variability of SGD.
3. Perform output perturbation on saved models, and **test private model** performance.
4. **Visualise** and report analyses.

# Codebase components

- `wrapper.py` is a wrapper for steps in the workflow, providing a CLI interface

Pertaining mostly to **running a sweep**:
- `run_experiment.py` provides the logic for reading configs, and running an experiment through loading data through `data_utils.py` and defining a model through `model_utils.py`
- `data_utils.py` contains all logic around loading and preprocessing training datasets.
- `model_utils.py` contains model specifications (e.g. CNN, logistic regression), including training loop and logic for saving intermediate weights

Concerned with **analysing the results** of the sweep:
- `derived_results.py` defines the set of derived results we compute on the output of a sweep and the logic for computing them. It calls on `results_utils.py` to interface with the results files. It also calls on `test_private_model` to compute derived results for utility.
- `results_utils.py` contains the logic for reading and loading experiment results, e.g. the weights of a given model.
- `test_private_model.py` implements computing the correct level of noise for a model (given derived results) and performing output perturbation, then computing private model performance.

There are also various functions for **visualisation and reporting**:
- `analyse.py` needs refactoring (at time of writing 8/6/20) but aims to contain the logic for generating the high-level results and figures for reporting in a paper.
- `vis_utils.py` contains visualisation functions that are either called by `analyse` or stand on their own for debugging and exploration.

Other miscellaneous files:
- `experiment_metadata.py` stores metadata like the number of weights in a model, the convergence points we have computed for different experiments, colours for plotting etc. This requires manual updating as needed.
- `stats_utils.py` is largely vestigial but was aimed at fitting different distributions, mostly to the gradient noise.

# Worked Example

An example on the binary-task (0 v 1) version of MNIST using logistic regression.
We assume there will be a config called `mnist_binary`.

## Run experiments

To run a single experiment with `seed = 5` and `replace_index = 10` we would use

```python run_experiment.py --cfg mnist_binary --seed 5 --replace_index 10```

without specifying the seed and replace_index, it will use `1` and `None` respectively.

To run a sweep of experiments using a grid of 25 seeds and 30 replace indices, we use

```python wrapper.py sweep --cfg mnist_binary --num_seeds 25 --num_replaces 30```

This runs each seed and replace_index configuration twice - once with a fixed initialisation and once with variable.
In total then, it will run 25 * 30 *2 = 1500 experiments in serial.

## Compute derived results

```python wrapper.py derive --cfg mnist_binary --t 2000```

This will compute the suite of derived results (see [Derived Results](#derived-results)) at `2000` training steps (where the derived result requires this input). If `t` is not provided, it will first attempt to compute the convergence point of the setting to use as `t`.


## Visualise

To generate the figures corresponding to the `mnist_binary` experiment:

```python wrapper.py figures --cfg mnist_binary --t 2000```

This calls `generate_plots` from `produce_figures.py`, which visualises:
- the delta histogram
- the distribution of epsilon values
- the sensitivity and variability over time
- the stability of estimated values

## Report

To compute specific (non-figure) values, such as the performance of the trained model:

```python wrapper.py report --cfg mnist_binary --t 2000```

This calls `generate_reports` from `produce_figures.py`, which reports:
- empirical and theoretical sensitivity
- empirical intrinsic sigma for fixed and variable initialisation
- delta
- intrinsic epsilon using theoretical and empirical sensitivity
- performance at epsilon = 1 and 0.5 for private models

# Derived Results 

This section describes the set of 'derived results' that we compute. In code we subclass a `DerivedResult` object defined in `derived_results.py`.
Because some of these computations can be involved and their outputs can be used in several ways, we assume that we generate a derived result once (if possible) and simply load it in future analyses.

The substance is in the `generate` method.

Derived results typically live in a `/derived` subfolder in the trace folder for that experiment, e.g. the derived results for logistic regression on binary MNIST live in 
```/bigdata/traces_aDPSGD/mnist_binary/all/logistic/derived```.

### `DeltaHistogram`
We compute the distribution of Δ values varying either random seed, replace_index, or both (keeping the other fixed when appropriate).
This is saved as a dictionary with 3*2 keys: two for each of the variation options above - we save a list of which experiment pairs were used (identified by their seed and replace_index), and the resulting delta value.
The output of this is used to estimate the empirical sensitivity.

Reminder: Δ is the l2-distance between the weights of a pair of model.

### `UtilityCurve`
We compute the performance of a private model as a function of epsilon, for various values of epsilon not greater than 1. (TODO fix this in code).
For each trained model tested, four variants are saved:
- noiseless - no output perturbation
- `bolton` - following Wu et al.
- `augment` - proposed method, using fixed initialisation
- `augment_diffinit` - proposed method, variable initialisation (this is the default that we report)

### `AggregatedLoss`

Since we have a loss curve for each individual experiment, this produces an aggregated version by computing the mean and standard deviation of the loss as a function of training steps.
Mostly used for visualisation.

### `SensVar`

This is about computing pairwise sensitivity and variability across experiment pairs - used later to visualise the distribution of pairwise epsilon.

### `Sigmas`

This is about estimating the intrinsic σ given a fixed `replace_index`, i.e. we load the posterior (from different seeds) of an experiment at a fixed time-point and estimate σ.
We save this estimate for a number of `replace_index` values, so that by later loading and selecting the minimum, we can report the overall estimated variability.

### `VersusTime`

This is something of a wrapper around other `DerivedResult` objects - the objective is to compute the estimated sensitivity and variability as a function of training time, so it needs to repeatedly use results from `DeltaHistogram` and `Sigmas`.

### `Stability`

This aims to understand how 'stable' our estimates are as a function of the number of experiments or seeds we used to estimate them, so it also repeatedly uses `DeltaHistogram` and `Sigmas`.

# Config files

Experiment settings are managed using configs stored in yaml files.
For binary MNIST, it looks like this:
```
---
data:
    name: mnist
    binary: true
    flatten: true
    preprocessing: GRP
model:
    architecture: logistic
    task_type: binary
    input_size: 50
    output_size: 1
    hidden_size: 10      # doesnt make sense for logistic
training:
    n_epochs: 2            # should be 20
    batch_size: 32
    optimization_algorithm:
        name: SGD
        learning_rate: 0.5
logging:
    cadence: 50
    save_weights: true
    save_gradients: true
    sample_minibatch_gradients: false
    n_gradients: 0
...
```
- In `data` we define preprocessing steps as needed - these will either result in a new version of the dataset being created, or loading a pre-processed version. _(GRP means Gaussian Random Projections, which is how we flatten MNIST images to 50-dimensional vectors in this example)_.
- In `model` we define the model architecture and task type (since some architectures can do multiple tasks) and in/out sizes, as well as architecture-specific hyperparameters.
- In `training` we define training hyperparameters such as batch size and the maximum number of epochs. We also define optimizer settings, although at present only SGD is supported.
- In `logging` we define how much saving we will do during training - this is important since most of the analysis is of these artefacts saved during training. The cadence sets how often (in batches) we save. Sampling minibatch gradients is currently not supported, but used to be for estimating gradient statistics at a fixed point in the parameter space (e.g. without taking gradient steps, simply sampling many gradients and storing them).

The config file does _not_ specify the `seed`, `replace_index`, or `diffinit` values because we typically sweep across these. What is specified in the config should be true for all experiments we run given that data+model choice.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
