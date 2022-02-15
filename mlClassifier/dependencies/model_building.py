from utils.data_preprocessing import DataCleaning, DataReshaping
from utils.data_preprocessing import encode_normalise
from tensorflow import keras as k

features_to_use = ['Sha256', 'isPeFile', 'isValidSignedFile', 'fileSize',
                   'ImportFunctionCount', 'ImportModuleCount', 'PeAppendedSize']

"""
## add training and testing data in root for model building (model data folder).
"""

data_cleaning = DataCleaning('../model_data/train_sample.csv', '../model_data/test_sample.csv')
data_cleaning.import_data()
[train_sample, test_sample] = data_cleaning.remove_null_values()

train_sample = encode_normalise(train_sample)
test_sample = encode_normalise(test_sample)

"""
## reshaping samples.
"""

data_reshape = DataReshaping(train_sample, test_sample, features_to_use)
data_reshape.select_features()
target = data_reshape.get_target_variable()
[train_sample, test_sample] = data_reshape.force_dataframe_resize()

"""
## Adding back the target data.
"""

train_sample['label'] = target

"""
## Using keras sequential model with at least 3N and sigmoid activation function.
"""

model = k.Sequential()

"""
## need to define NN architecture.
"""