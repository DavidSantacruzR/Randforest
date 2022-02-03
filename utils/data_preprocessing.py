import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataCleaning:

    """
    ## Class constructor
    """

    def __init__(self, train_sample_name: str, test_sample_name: str):
        self.train = train_sample_name
        self.test = test_sample_name
        self.train_label = train_sample_name
        self.test_label = test_sample_name

    """
    ## currently importing data from csv and dropping NA values in both datasets.
    """

    def import_data(self):
        self.train = pd.DataFrame(pd.read_csv(self.train, delimiter=';'))
        self.test = pd.DataFrame(pd.read_csv(self.test, delimiter=';'))

    def remove_null_values(self):
        self.train = pd.DataFrame(self.train).dropna()
        self.test = pd.DataFrame(self.test).dropna()

        return [self.train, self.test]

    """
    ## Encoding text variables using a simple labeler encoder.
    ## Normalising variables using standard scaler z = x-mean/ste.dev.
    Not recommended with heavy outliers in the dataset.
    """

    def encode_normalise(self, training_sample, testing_sample):
        pass


class DataReshaping:

    """
    ## Constructor selecting the features vector excluding the target.
    ## Pending target reshaping.
    """
    def __init__(self, sample_1, sample_2, features_vector: list):
        self.x1 = pd.DataFrame(sample_1)
        self.x2 = pd.DataFrame(sample_2)
        self.variables_vector = features_vector

    def get_target_variable(self):
        max_columns: int = len(self.x1.columns) - 1
        target = self.x1.pop(self.x1.columns[max_columns])

        if self.x1.shape[0] > self.x2.shape[0]:
            max_rows: int = self.x2.shape[0]
            target = pd.DataFrame(target).iloc[0:max_rows]
            return target

    def select_features(self):
        return [self.x1[self.variables_vector], self.x2[self.variables_vector]]

    def force_dataframe_resize(self, target):

        if self.x1.shape[0] < self.x2.shape[0]:
            max_rows: int = self.x1.shape[0]
            self.x2 = pd.DataFrame(self.x2.iloc[0:max_rows])
            return [self.x1, self.x2]

        else:
            max_rows: int = self.x2.shape[0]
            self.x1 = pd.DataFrame(self.x1.iloc[0:max_rows])
            return [self.x1, self.x2]