# Beyond the Full Slate: Evaluating MNL Algorithms on All Subsets

This repository contains the code to reproduce the experiments in the paper "Beyond the Full Slate: Evaluating MNL Algorithms on All Subsets". The implementation of our algorithms to compute the metrics are in the directory "evaluate". 

The implementation of some non-adaptive algorithms is taken from choix (https://github.com/lucasmaystre/choix). We used their entire code for ILSR and MM rather than calling the functions directly from the library as we required to make some changes. 

## Running the experiments
The following command runs the full set of experiments. Running all the experiments required days of computation on a standard desktop computer. By inspecting the main function of run_experiments.py, it is possible to exclude some datasets that are quite demanding (e.g., Codeforces).

```bash
python3 -m experiments.run_experiments
```

## Plotting the results

The following command plots several errors metrics for each algorithm on each dataset:
```bash
python3 -m plotting.plot_results
```

The following command plots the performance of non-adaptive algorithms against the adaptive one:
```bash
python3 -m plotting.plot_algo_comparison
```

The following command plots the results of our pair distribution experiment:
```bash
python3 -m plotting.plot_local_pairs_distribution
```

The following command plots the results of our sparsification experiments:
```bash
python3 -m plotting.plot_skip_datasets
```

The following command plots the results of our synthetic experiments with a power-law distribution:
```bash
python3 -m plotting.plot_powerlaw_datasets
```
