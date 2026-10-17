# LSTM_py

## About

LSTM_py provides Python scripts for LSTM hyperparameter search, model training,
and saved-model prediction for activated-sludge-model (ASM) surrogate workflows.

## Scripts

- `lstm_hypersearch.py`: searches candidate LSTM configurations;
- `lstm_training.py`: trains and saves selected models; and
- `load_and_predict.py`: loads a trained model and produces predictions.

## Use

Create an isolated Python environment, inspect each script for its expected
input columns and local paths, install the imported dependencies, and run the
desired stage. Keep data preprocessing and temporal train/validation/test
splits consistent across all three stages.

## Status

These scripts are research prototypes rather than a packaged library. Validate
units, scaling, sequence length, and out-of-sample performance before reuse.
