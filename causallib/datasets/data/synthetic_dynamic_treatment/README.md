# Synthetic data for dynaminc treatment

Data is generated according to 

$$ A \sim Binomial(\text{invLogit}(X1_{t-1} - \bar{x1})/10 - A_{t-1}) $$

$$ X1 \sim Normal(A_t + X1_{t-1}, 1) $$  

$$ X2 \sim Normal(0, 1) $$  

Data contains 5 attributes: `id, time, treatment (A), cov1 (X1), cov2 (X2)`

Under this observational strategy, the simulated files are generated using the following command:
```bash
python mk_synthetic_data.py
```
This produces $4$ files
- `./simx2_observational.csv` : This the original file containing data simulated under observational strategy
- `./simx2_observational_train.csv`: training fold from the original data. Stratified according to total treatment for each individual
- `./simx2_observational_val.csv`: validation fold from the original data. Stratified according to total treatment for each individual
- `./simx2_observational_test.csv`: test fold from the original data. Stratified according to total treatment for each individual

In addition, a dataset for $100$ individuals under the `never treat` (i.e. $A = 0, \forall t$) counterfactual strategy is also generated using the following command
```bash
python mk_synthetic_data.py -c -n 100
```


## Shifted dataset

A shifted dataset can be generated with previous values of past covariates with covariates 
structured as below
- independent: `id, time, prev_treatment, prev_cov1, prev_cov2`
- dependent: `treatment` (boolean), `prev_cov1` (float), `prev_cov2` (float)

To generate these files, use the previous command with the `--shifted` flag


## Full Usage

```bash
usage: mk_synthetic_data.py [-h] [-s SEED] [-t NUM_TIMEPOINTS] [-n NUM_SAMPLES] [-c] [--shifted] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  seed value to sample the data. Default=20
  -t NUM_TIMEPOINTS, --num_timepoints NUM_TIMEPOINTS
                        Number of timepoints to sample from. Default=50
  -n NUM_SAMPLES, --num_samples NUM_SAMPLES
                        Number of samples. Default=1000
  -c, --counterfactual  Strategy to use. if provided use 'never-treat' else uses 'observational'
  --shifted             if provided, also generate shifted samples of chosen strategy
  -v, --verbose         Log option: Write debug messages.
```
