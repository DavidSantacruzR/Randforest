from utils import data_preprocessing
from mlClassifier.dependencies.logistic_regression_classification import ModelBuilding, PredictionService
import pandas as pd

"""
## creates an instance of the preprocessing class, which imports data and drops null values.
## assigns each clean sample to their respective dataframes.
"""

first_instance = data_preprocessing.DataPreProcessing('train_sample.csv', 'test_sample.csv')
first_instance.import_data()
[train_sample, test_sample] = first_instance.remove_null_values()

# Here goes the encoding and normalising data.

"""
## each dataset should have the same features (same name references).
"""

second_instance = data_preprocessing.FeatureSelection(train_sample, test_sample)
[train_sample, test_sample] = second_instance.select_features()

train_sample = pd.DataFrame(train_sample)
test_sample = pd.DataFrame(test_sample)

"""
## once having the data then we need to get the model built.
"""

third_instance = ModelBuilding(train_sample, test_sample)
third_instance.force_equal_split()
x_train, x_test, y_train, y_test = third_instance.data_preparation()


fourth_instance = PredictionService()  # Not yet ready.

print('x_train:', x_train.shape,
      '\n x_test:', x_test.shape,
      '\n y_train:', y_train.shape,
      '\n y_test:', y_test.shape)

"""
## Notes to self:
## resizing using the array method does not work properly. Removes column labels and other relevant data.
"""
