(section:edgeattgcn_validation)=
# Hyperparameter tuning for Edge-Enhanced Attention GCN

## Model validation

In order to validate our custom layer, several checks were performed:

* Setting the edge feature matrix $E^l$ to ones
* Setting the edge-enhanced attention matrix $A^l$ to ones before multiplying with the support
* Removing the normalization

The results can be seen here :

<iframe src="https://wandb.ai/c-achard/DL%20Biomed%20Homework%202%20-%20v3/reports/Validation-of-Edge-Enhanced-Attention-GCN--Vmlldzo1NzM4MDQy" style="border:none;height:1024px;width:100%"> </iframe>

## Global hyperparameters

### Hyperparameter tuning

Finally, several variants were tuned for the EGNN(A).

Overall, integrating edge features improved performance, however the model required adjustments to the learning rate and number of epochs to achieve the best results found here.

The tuning was mostly done via layer architecture and learning rate here, as changing the previously tuned hyperparameters mostly showed the same patterns.
It was found that decreasing the step size by a factor of 10 increased performance; however the previously used layer architecture generally enabled better performance.

Also, this model converges much faster than the previous ones, and achieves a higher performance on the test set;
this offsets the fact that the overall speed of computation is decreased due to the additional parameters.

Early stopping at 200 epochs was used for the best run, as it was found that slight overfitting on the train set tended to occur when allowing the model to train for longer. This longer run is not shown to keep more comparable curves for other runs.

<iframe src="https://wandb.ai/c-achard/DL%20Biomed%20Homework%202%20-%20v3/reports/Edge-Enhanced-Attention-GNN-Tuning--Vmlldzo1NzM2NTYw" style="border:none;height:1024px;width:100%"> </iframe>

### Comparison with using node features only

We can observe that the additional features provided by edges do help performance when used in our custom architecture.

```{note}
To visualize and compare performance of all models, please see {ref}`section:best_runs`.
```

However we should keep in mind the relative smallness of the dataset; indeed adding more features might allow us to overfit on the whole MUTAG dataset, but might not generalize well to new, outside-of-dataset molecules.
A more careful analysis of the types of molecules in the dataset and how they were acquired might be necessary to determine if this is the case; we could also focus on interpretability to see if the model is learning meaningful features overall.

## Links to reports

1) [Validation of Edge-Enhanced Attention GCN](https://api.wandb.ai/links/c-achard/9d6woq39)
2) [Edge-Enhanced Attention GCN Tuning](https://api.wandb.ai/links/c-achard/hhwoq3eg)
