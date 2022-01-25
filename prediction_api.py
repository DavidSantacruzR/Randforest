from utils import data_preprocessing
from mlClassifier.dependencies.logistic_model_preprocessing import ModelPreProcessing, ModelScalingAndEncoding
import pandas as pd

"""
## creates an instance of the preprocessing class, which imports data and drops null values.
## assigns each clean sample to their respective dataframes.
"""

first_instance = data_preprocessing.DataPreProcessing('train_sample.csv', 'test_sample.csv')
first_instance.import_data()
[train_sample, test_sample] = first_instance.remove_null_values()

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

third_instance = ModelPreProcessing(train_sample, test_sample)
third_instance.force_equal_split()
x_train, x_test, y_train, y_test = third_instance.data_preparation()

fourth_instance = ModelScalingAndEncoding(x_train, x_test, y_train, y_test)
fourth_instance.feature_scaling()  # defines, selects and scales a variable if needed.
fourth_instance.feature_encoding()  # defines, selects and encodes text variables in the sample.
# Encode required text data, either binary or numerical.

"""
## Notes to self:
## resizing using the array method does not work properly. Removes column labels and other relevant data.
"""
