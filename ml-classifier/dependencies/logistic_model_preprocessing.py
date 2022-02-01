import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_preprocessing import DataPreProcessing


class ModelPreProcessing:
    def __init__(self, x_sample, y_sample):
        self.x_sample = pd.DataFrame(x_sample)
        self.y_sample = pd.DataFrame(y_sample)
        self.target_label = len(list(y_sample.columns))

    def get_target_label(self):
        self.x_sample.pop(self.target_label)

    def force_equal_split(self):
        """
        ## requires the data to be the same shape before using train_test_split as it expect
        ## objects to be the same size.
        :return:
        """

        self.x_sample, self.y_sample = DataPreProcessing(self.x_sample, self.y_sample).resize_dataframe()

        return print(self.x_sample.shape[0], self.y_sample.shape[0])

    def data_preparation(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_sample,
            self.y_sample,
            test_size=0.33,
            random_state=42)

        return [x_train, x_test, y_train, y_test]


class ModelScalingAndEncoding:
    def __init__(self, x_training_sample, x_testing_sample, y_training_sample, y_testing_sample):
        self.x_train = x_training_sample
        self.x_test = x_testing_sample
        self.y_train = y_training_sample
        self.y_test = y_testing_sample

    def feature_encoding(self):
        """
        ## Encoding true/false variables as [0,1] binaries.
        ## The first variable in the dataset is excluded as it's supposed to be the file labels.
        """
        pass

    def feature_scaling(self):
        """
        ## Uses the MinMaxScaler to scale variable values in the range [0, 1]
        ## The first variable in the dataset is excluded as it's supposed to be the file labels.
        """
        pass
