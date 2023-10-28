# DeepLearningBiomedicine-Homework2

Deep learning homework on graph classification
*Author : Cyril Achard - 310048*

## Project structure

The project directory is structured as follows:

* [WEBSITE](https://c-achard.github.io/DeepLearningBiomedicine-Homework2/intro.html) : The built **Jupyter Book**, containing the notebooks and analysis
* `report` : contains the **PDF report**
* `book` : Contains the Jupyter Book source data and all notebooks/code
  * `code` : **Source code** of the project as .py files
    * `model.py` : Contains the models code (layers, heads, aggregators, etc.)
    * `training.py` : Contains the training code (training loop, evaluation, etc.)
    * `utils.py` : Contains data-loading/pre-processing and plotting utilities
  * `rendered_notebooks` : Rendered **notebooks** of best runs
  * `wandb_comparisons` : HTML reports of the **hyperparameter tuning**

## Jupyter Book structure

This repository contains the Homework 2 of the Deep Learning in Biomedicine course.

* The first section introduces the dataset and the preprocessing used.
* The second section contains reports of hyperparameter tuning for the different models.

```{warning}
The reports are embedded html reports from Weights and Biases.
If you encounter any problem, please use the provided links to access the reports at the bottom of each section instead.
```

* The third and last section contains the notebooks with the best run of each model executed.

## Tools used

* Hyperparameter tuning and interactive plots with Weights and Biases
* Models are pure PyTorch (geometric was not used, even for data loading)
* Data loading with HuggingFace datasets
* Graphs visualisation with NetworkX
* Documentation and structured notebooks with Jupyter Book
* Report with Overleaf
* Code formatting with pre-commit and ruff (w/ black and isort)
