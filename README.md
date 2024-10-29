# Neuroanatomical Model Processing and Visualization

This module processes neuroanatomical sulcal data using bootstrap sampling and piecewise linear models, and generates visualizations. It can analyze features like sulcal opening or cortical thickness across different datasets, atlases, and percentiles.

## Table of Contents
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
- [Functions](#functions)
  - [read_sulci](#read_sulci)
  - [bootstrap](#bootstrap)
  - [piecewise_linear_percentile](#piecewise_linear_percentile)
  - [display_linear_bootstrap](#display_linear_bootstrap)
  - [slope_change](#slope_change)
  - [extract_models_from_csv](#extract_models_from_csv)
  - [save_coeff](#save_coeff)
  - [process](#process)
- [Examples](#examples)

## Installation

This module requires Python 3 and the following packages:
- `numpy`
- `pandas`
- `matplotlib`

Install dependencies with:
```bash
pip install numpy pandas matplotlib
```

## Pipeline Overview
The pipeline allows for:
1. **Data reading** (`read_sulci`): Loads the dataset based on feature, atlas, and dataset type.
2. **Bootstrap Sampling** (`bootstrap`): Resamples data to produce multiple training sets.
3. **Piecewise Model Fitting** (`piecewise_linear_percentile`): Fits a model to specified percentiles of the data.
4. **Visualization** (`display_linear_bootstrap`): Plots the model fits.
5. **Result Saving** (`save_coeff`): Saves computed model parameters in a Morphologist-compatible format.

## Usage
The `process` function initiates the entire pipeline, handling data preprocessing, model fitting, and saving outputs for further neuroanatomical analysis.

## Functions

### `read_sulci(feature, atlas, dataset)`
Loads the neuroanatomical dataset by feature, atlas, and dataset type.

**Parameters**
- `feature` (str): The feature to extract, e.g., `'opening'` or `'thickness'`.
- `atlas` (str): The brain parcellation atlas, e.g., `'brainvisa'` or `'desikan'`.
- `dataset` (str): Dataset name, e.g., `'UKB'` or `'Memento'`.

### `bootstrap(data, feature, sulci_names, foldername, proportion, n_bootstraps, percentile)`
Generates bootstrap samples for data and saves models.

**Parameters**
- `data` (DataFrame): The dataset to bootstrap.
- `feature` (str): Feature of interest, e.g., `'opening'` or `'thickness'`.
- `sulci_names` (list of str): Names of ROIs.
- `foldername` (str): Folder path for saving files.
- `proportion` (float): Proportion of data in each bootstrap sample.
- `n_bootstraps` (int): Number of bootstrap samples.
- `percentile` (int): Quantile line between 0 and 100.

### `piecewise_linear_percentile(data, percentile)`
Fits a piecewise linear model for a specified percentile.

**Parameters**
- `data` (DataFrame): Input data.
- `percentile` (int): Quantile line between 0 and 100.

### `display_linear_bootstrap(data, tabs, data_bootstrap, foldername, feature, atlas, percentile, gender, show)`
Plots the piecewise linear model fit on bootstrap samples.

**Parameters**
- `data` (DataFrame): Input data.
- `tabs` (dict): Model parameters for each bootstrap iteration.
- `data_bootstrap` (dict): Training data for each bootstrap iteration.
- `foldername` (str): Folder path for saving plots.
- `feature` (str): Feature name.
- `atlas` (str): Atlas name.
- `percentile` (int): Quantile line.
- `gender` (str): Gender designation, e.g., `'F'` or `'M'`.
- `show` (bool): Display the plot if `True`.

### `slope_change(a1, a2)`
Calculates the change in slope between model segments.

**Parameters**
- `a1` (float): Slope of the first line segment.
- `a2` (float): Slope of the second line segment.

### `extract_models_from_csv(sulci_names, n_bootstraps, foldername)`
Retrieves model parameters and training data from saved files.

**Parameters**
- `sulci_names` (list of str): ROIs for which models were fitted.
- `n_bootstraps` (int): Number of bootstrap samples.
- `foldername` (str): Folder path with saved `.txt` and `.npy` files.

### `save_coeff(col_names, sulci_names_, coeff, filename)`
Saves model parameters in a CSV format compatible with Morphologist.

**Parameters**
- `col_names` (list of str): Model parameter names, e.g., `['a1', 'b1', ...]`.
- `sulci_names_` (list of str): Sulci names.
- `coeff` (array): Model parameters for each sulcus.
- `filename` (str): Path for the CSV file.

### `process(output_dir, feature, dataset, atlas, n_bootstraps, percentile, gender=None)`
Runs the pipeline on a dataset, with specified bootstraps and percentiles.

**Parameters**
- `output_dir` (str): Output folder path.
- `feature` (str): Feature, e.g., `'opening'` or `'thickness'`.
- `dataset` (str): Dataset name, e.g., `'UKB'` or `'Memento'`.
- `atlas` (str): Atlas, e.g., `'brainvisa'` or `'desikan'`.
- `n_bootstraps` (int): Number of bootstrap samples.
- `percentile` (int or list of int): Quantile lines.
- `gender` (str, optional): Gender, e.g., `'F'` or `'M'`.

## Examples

Run the `process` function to apply the entire pipeline. 

```python
# Example 1: Analyze the 90th percentile of Brainvisa sulcal opening on UKB women
process(output_dir=".", feature="opening", dataset="UKB", atlas="brainvisa", n_bootstraps=2, percentile=90, gender="F")

# Example 2: Analyze the 80th and 45th percentiles of Desikan thickness on UKB men
process(output_dir=".", feature="thickness", dataset="UKB", atlas="desikan", n_bootstraps=2, percentile=[80, 45], gender="M")
```

