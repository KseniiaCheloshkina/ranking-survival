## Ranking Weibull Survival Model: boosting concordance index of Weibull time-to-event prediction model with ranking losses

This repository provides an implementation of methods and experiments results. 

### Repository structure
Models:
- `train.py` - script for training a model for a specified dataset
- `hp_search.py` - script for grid search over hyper-parameters of models  

Reproduction of results:
- `eval.py` - script to reproduce results presented in paper 
- `benchmark.py` - script to reproduce benchmark models performance presented in the paper

Data for experiments:
- datasets for experiments (METABRIC and KKBOX) were downloaded from `pycox` Python package (methods `pycox.datasets.metabric.read_df()` and `pycox.datasets.kkbox_v1.read_df(subset)`)
- all preprocessing is described in `save_preprocessed_data.py`. The input for the script is a result of calling methods `pycox.datasets.metabric.read_df()` and `pycox.datasets.kkbox_v1.read_df(subset)`. Output is presented by preprocessed datasets which are saved as `pickle` files in folders `data/input_metabric` and `data/input_kkbox` respectively.
- configs for models are stored in `configs` folder and used in `eval.py`
- `custom_models.py` - contains feed-forward networks for the datasets

The models itself are described in the next files:
- `batch_generators_hard_mining.py` - contains online batch generator
- `losses.py` - contains different losses functions
- `models_hard_mining.py` - contains models description
- `tools.py` - contains evaluation and preprocessing functions
