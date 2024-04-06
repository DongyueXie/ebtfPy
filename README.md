## Empirical Bayes Trend Filtering 

This repo has code and analysis for the Empirical Bayes Trend Filtering method. To create environment: `conda env create -f requirements.yaml`

## Code

Source code is in the `src` folder, and simulation code is in `VEBTF-paper/simulation`. 

## Analysis

To re-produce the results in the paper: for simulation, run the `VEBTF-paper/simulation/run_simulation.sh`, then use `VEBTF-paper/simulation/notebook/analyze_simu.ipynb` for getting metrics.

For real dataset, the accelaration example is in `src/accel.ipynb`; for the rest, clone the repo `https://github.com/alan-turing-institute/TCPD/tree/master` and obtain the dataset following its instructions, then run the `VEBTF-paper/simulation/realdata/dataset.ipynb` in the directory `TCPD/examples/python`

The figures related to sparse EBTF is in `src/test_sparse.ipynb`