## Empirical Bayes Trend Filtering

This repository contains code and analysis for the Empirical Bayes Trend Filtering (EBTF) method.

### Environment Setup
To set up the environment, run:
```bash
conda env create -f requirements.yaml
```
This repository also requires several R packages, which can be installed using the notebook:
```
VEBTF/notebook/install_r.ipynb
```

## Code Structure
- **EBTF Source Code**: Located in the `src` folder.
- **Simulation Code**: Found in `VEBTF-paper/simulation`.
- **Real Data Analysis Code**: Stored in `VEBTF-paper/realdata`.

## Analysis
### Reproducing Paper Results

#### Simulation
1. Navigate to the `VEBTF-paper` directory.
2. Run the simulation script:
   ```bash
   ./simulation/run_simulation.sh
   ```
3. Analyze simulation results using the notebook:
   ```
   VEBTF-paper/notebook/analyze_simu.ipynb
   ```

#### Real Data Analysis
- **Acceleration example**: `src/accel.ipynb`
- **ETTh1 dataset**: `realdata/real_data_etth1.ipynb`
- **Illness dataset**: `realdata/real_data_illness.ipynb`
- **Weather dataset**: `realdata/real_data_weather.ipynb`
- **Holdout evaluation**: Navigate to the `VEBTF-paper` directory,
  ```bash
  ./realdata/run_realdata.sh
  ```

### Sparse EBTF Figures
Figures related to sparse EBTF are generated using:
```
src/test_sparse.ipynb
