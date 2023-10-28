(section:tuning)=
# Hyperparameter tuning for GCN

## Global hyperparameters

Please find below a comparison of runs with different global hyperparameters.
The hyperparameters were:

* Activation function
* Initialisation scheme
* Pooling method
* Loss weighting

```{note}
Due to an issue with reproducibility, the run showcasing performance with loss weighting was not included in the comparison.
This run was appearing as being the best, but as performance could not be exactly replicated, it was decided to omit it.
It is still visible in the list of wandb runs, for your curiosity.
Also note that not all runs performed are shown, only meaningful ones that hepled the hyperparameter tuning were kept in order to keep some clarity.
```

After analysis, it was decided to use :

* SELU or LeakyReLU : Negative features contributed to performance when used, therefore ReLU usage was discontinued.
* Max pooling: Instead of smoothing (mean) which consistently hindered performance, pooling was done according to the most salient features (max) before the classifier head.
* Kaiming initialization : An initialization scheme that does take into account non-linearities enabled better performance than Glorot or default initialization by a small margin.
* Loss weigthing, to account for the moderate class imbalance, did improve performance slightly.

These were used as a baseline for the following models; changes were still made where necessary.

```{hint}
If you wish to show/hide certain runs for clarity, please scroll down to the bottom of the report and use the **Run set** table.
```

<iframe src="https://wandb.ai/c-achard/DL%20Biomed%20Homework%202%20-%20v3/reports/Comparison-of-GCN-performance--Vmlldzo1NzMxOTYz" style="border:none;height:1024px;width:100%"> </iframe>

## Layer hyperparameters

Here, several layer numbers and shapes were compared.

The architecture using 4 layers of shape $256,256,128,64$ consistently performed best, and was retained.


<iframe src="https://wandb.ai/c-achard/DL%20Biomed%20Homework%202%20-%20v3/reports/Comparison-of-GCN-with-different-layers--Vmlldzo1NzM2MTU4" style="border:none;height:1024px;width:100%"> </iframe>


## Links to reports

1) [Link to first GCN report](https://api.wandb.ai/links/c-achard/2bgiali8)
2) [Link to second GCN report](https://api.wandb.ai/links/c-achard/00m6utth)
