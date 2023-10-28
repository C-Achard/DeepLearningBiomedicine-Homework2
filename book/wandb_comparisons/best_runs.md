(section:best_runs)=
# Summary of model performance

## Best performance summary

Below is the table of the best runs for this homework, per model.

```{list-table}
:header-rows: 1
* - Model
  - Test AP
  - Test ROC-AUC
* - *GCN*
  - 0.959
  - 0.888
* - *GraphSAGE*
  - 0.953
  - 0.877
* - *AttentionGCN*
  - 0.937
  - 0.872
* - ***EdgeAttentionGCN***
  - **0.9649**
  - **0.911**
```

## Conclusion and possible extensions

With this homework we have seen how to use graph neural networks to classify nodes in a graph, and how to use the attention mechanism to improve the performance of the model.
In addition, we show that a custom model making use of edge features can improve the performance of the model.

However, due to the small size of the dataset, it is difficult to conclude that the model is learning meaningful features, and that it would generalize well to completely new molecules outside of the MUTAG dataset. A more detailed analysis could help disentangle this.

An interesting addition could have been to focus on interpretability of the models, and to check the gradient of each class with respect to the input features, to see which features were most important in the predictions, à la GradCAM.

One could for example adapt the approach suggested in [GCNN-Explainability](https://github.com/ndey96/GCNN-Explainability), based on Pope et al., 2019.

## Comparison of best runs

Below is an interactive comparison of best runs. Should it not work as intended, please use the link provided in the next section.

<iframe src="https://wandb.ai/c-achard/DL%20Biomed%20Homework%202%20-%20v3/reports/Homework-2-Best-Runs--Vmlldzo1NzM2NTgy" style="border:none;height:1024px;width:100%"> </iframe>

### Link to reports

Please find here an interactive view from the [WandB report of best runs](https://api.wandb.ai/links/c-achard/flrx078r)
