## Installation

Run the following to install:

```python
pip install lorenz_zonoid
````

```python
from lorenz_zonoid import ShapleyLorenzShare

# Compute Shapley Lorenz Zonoid values:
lorenzshare = ShapleyLorenzShare(model.predict, X_background_data, y_background_data)
lorenzshare.shapleyLorenz_val(X_testset)
```

# Developing Shapley Lorenz Zonoid

To install lorenz_zonoid, along with the tools you need to develop and run tests, run the following in your virtualenv:

```bash
$ pip install -e .[dev]
```
