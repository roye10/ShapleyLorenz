# ShapleyLorenz

!![SLZ](Pictures/logo.png)

## Installation
this package can be installed from [PyPi](https://pypi.org/project/shapley-lz/) using the following command

```
pip install shapley_lz
```

## Summary

Algorithm that computes Shapley-Lorenz contribution coefficients, as defined in the paper "Shapley-Lorenz decompositions in eXplainable Artificial Intelligence", by Paolo Giudici and Emanuela Raffinetti from February 2020.

The function takes as input
* the pre-trained model `f(·)`, which is to be explained,
* a of the training covariance matrix `X_train` and
* a covariance test set, `X_test`, whose output, `f(X_test)` is to be explained
and returns an array of Lorenz Zonoid value for each feature, computed using the Shapley attribution mechanism, in order to account for interaction effects.

## Exmaple Using a Random Forest Classifier
```Python
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf_class
from sklearn.datasets import make_classifaction as gen_data

# Simple example w/o train-test splitting thus same covariance matrix used and only first 100 observations explained
N = 1000 # number of observations
p = 4 # number of features
X, y = gen_data(n_samples = N, n_features = 4, n_informative = 4)
model = rf_class()
model.fit(X,y)
slz = ShapleyLorenzShare(model.predict_proba, X, y)
slz_values = slz.shapleyLorenz_val(X[:100,:], y[:100], class_prob = True, pred_out = 'predict_proba')

# Plot
slz.slz_plots(slz_values[0])
```

## Intuition

Plot of Lorenz Curves for simulation data set with three features and normally distributed features and error term:

![Lorenz curve for feature 2](Pictures/Lorenz_Curve.png)

How to read:
The diagonal 90-degree line represents a model that has no input features and forms its prediction as the average over all outcomes. Thus, the furter away the Lorenz curve for a prediction model with p features is from the 90 degree line, the more of the variation in the observed response variable, the model is able to explain.

By a Lemma as mentioned in the aforementioned paper, the Lorenz Zonoid of a model with p-1 features is always smaller (in terms of surface area, calculated by the difference between the Lorenz Curve and its inverse), than the Lorenz Zonoid of a model with p features. This is exemplified in the graph, where it can be seen, that the set of points between the 90-degree line and the Lorenz curve for the prediction model excluding feature k is a subset of the points between the 90-degree Line and the Lorenz curve of the prediction model including feature k.
