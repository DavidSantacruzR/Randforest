## RandForest
### Statistical classifier using techniques based on random forests and decision trees

The **RandForest** library is a personal project aimed to provide a framework classify data between two or more classes
based on decision trees and different kernel algorithms.
---
### Usage:
#### Simple binary encoder
```python
from randforest.encoders import LabelEncoder

data = {'my_variable': [True, True, False, True, False]}

instance = LabelEncoder(data, ['my_variable'])
encoded_features = instance.fit_transform()
```

