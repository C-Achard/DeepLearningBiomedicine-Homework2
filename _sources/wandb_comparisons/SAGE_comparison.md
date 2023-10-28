# Hyperparameter tuning for GraphSAGE

## Aggregation functions comparison

Below is a comparison of runs with different aggregation functions.
The aggregation functions were:

* **SumAggregation**
* **MeanAggregation**
* **SquareRootAggregation**
* **LSTMAggregation**

Please see the {ref}`section:SAGE` notebook for more details on aggregation functions.

It was found that the LSTM aggregation function performed best, further tuning was therefore performed on this aggregation function, see below.

<iframe src="https://wandb.ai/c-achard/DL%20Biomed%20Homework%202%20-%20v3/reports/Comparison-of-SAGE-Aggregations--Vmlldzo1NzM2MTc5" style="border:none;height:1024px;width:100%"> </iframe>

## LSTM Aggregation fine-tuning

Below is a comparison of runs with different LSTM aggregation hyperparameters.

Dropout, learning rate and batch size were the main hyperparameters tuned;
it was found that increasing the batch size and number of epochs increased performance on the test set,
while changing the layers or increasing dropout did not.
Despite the model quickly fitting to the whole train set, training for additional epochs did increase performance on the test set, showing meaningful learning rather than overfitting.

Achieving the highest performance for this model does take a comparatively large number of epochs, however; we will see if integrating
edge features can help speed up the learning.

<iframe src="https://wandb.ai/c-achard/DL%20Biomed%20Homework%202%20-%20v3/reports/SAGE-LSTM-Tuning--Vmlldzo1NzM2NDcy" style="border:none;height:1024px;width:100%"> </iframe>

## Links to reports

1) [Link to first report](https://api.wandb.ai/links/c-achard/9ek2rx82)
2) [Link to second report](https://api.wandb.ai/links/c-achard/v9l9ber7)
