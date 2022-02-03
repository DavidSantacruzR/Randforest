from utils.data_preprocessing import DataCleaning, DataReshaping
from mlClassifier.dependencies.logistic_model_preprocessing import ModelBuilding, PredictionService

"""
## creates an instance of the preprocessing class, which imports data and drops null values.
## assigns each clean sample to their respective dataframes.
"""

features_to_use = ['Sha256', 'isPeFile', 'isValidSignedFile', 'fileSize', 'filePrevalence',
                   'GeoId', 'Extension', 'ImportFunctionCount', 'ImportModuleCount',
                   'PeAppendedSize', 'PeHeaderChecksum', 'PeTimestamp', 'FirstObserved']

data_cleaning = DataCleaning('train_sample.csv', 'test_sample.csv')
data_cleaning.import_data()
[train_sample, test_sample] = data_cleaning.remove_null_values()
data_reshape = DataReshaping(train_sample, test_sample, features_to_use)
data_reshape.select_features()
target = data_reshape.get_target_variable()
[train_sample, test_sample] = data_reshape.force_dataframe_resize(target)
predictions = PredictionService()

"""
## Adding back the target data.
"""

train_sample['label'] = target
train_sample.to_csv('example_data.csv')
