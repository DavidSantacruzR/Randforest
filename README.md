## RandForest
### Statistical classifier using techniques based on random forests and decision trees
***
The **RandForest** library is a personal project aimed to provide a framework classify data between two or more classes
based on decision trees and different kernel algorithms. Use it just for fun or learning.

### Usage:
#### Simple binary encoder
```python
from randforest.encoders import LabelEncoder

data = {'my_variable': [True, True, False, True, False]}

instance = LabelEncoder(data, ['my_variable'])
encoded_features = instance.fit_transform()
```

#### Linear regression model
```python
from randforest.models import LinearRegression

# Features used as explanatory variables in the linear model.
regressors = {
        'explanatory_1': [20, 30, 22, 18, 27],
        'explanatory_2': [200, 150, 120, 310, 280],
        'explanatory_3': [3, 3, 4, 5, 7]
    }

# The dependent variable on which predictions are going to be made.
target = {
        'target': [1, 1, 0, 0, 1]
    }

# Set of values used to perform predictions on the target.
to_predict = {
        'explanatory_1': [19, 24, 33],
        'explanatory_2': [180, 110, 204],
        'explanatory_3': [4, 6, 9]
    }

linear_model = LinearRegression(regressors, target, to_predict)
model_estimators = linear_model.transform_estimators_vector()
"""
The transform method, returns the estimator as a list: 
[ 0.09366783409371948, 0.00403455168376235, -0.17170184536614608, -1.691664155139609]
"""
```

