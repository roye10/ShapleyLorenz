# ShapleyLorenz

## Recently added
- extension of Kernel SHAP as proposed by Lundberg and Lee in "A Unified Approach to Interpreting Model Predicions", whereby SHAP values are 'standardised' according to their contribution to model accuracy

### - IN PROGRESS - 

Algorithm that computes Shapley-Lorenz contribution coefficients, inspired by the paper "Shapley-Lorenz decompositions in eXplainable Artificial Intelligence", by Paolo Giudici and Emanuela Raffinetti from February 2020.

Function takes in covariate matrix and response vector and outputs the Lorenz Zonoid shares of the specified features.

TODO:
- insert method for normalising features, so as to account for negative values

# Example plots

Plot of Lorenz Curves for simulation data set with three features and normally distributed features and error term:

![Lorenz curve for feature 2](Pictures/Lorenz_Curve.png)

How to read:
The diagonal 90-degree line represents a model that has no input features and forms its prediction as the average over all outcomes. Thus, the furter away the Lorenz curve for a prediction model with p features is from the 90 degree line, the more of the variation in the observed response variable, the model is able to explain.

By a Lemma as mentioned in the aforementioned paper, the Lorenz Zonoid of a model with p-1 features is always smaller (in terms of surface area, calculated by the difference between the Lorenz Curve and its inverse), than the Lorenz Zonoid of a model with p features. This is exemplified in the graph, where it can be seen, that the set of points between the 90-degree line and the Lorenz curve for the prediction model excluding feature k is a subset of the points between the 90-degree Line and the Lorenz curve of the prediction model including feature k.
