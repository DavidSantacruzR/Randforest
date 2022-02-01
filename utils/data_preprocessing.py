import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreProcessing:
    def __init__(self, train_sample_name: str, test_sample_name: str):
        self.train = train_sample_name
        self.test = test_sample_name

    def import_data(self):
        self.train = pd.DataFrame(pd.read_csv(self.train, delimiter=';'))
        self.test = pd.DataFrame(pd.read_csv(self.test, delimiter=';'))

    def remove_null_values(self):
        self.train = self.train.dropna()
        self.test = self.test.dropna()

        return [self.train, self.test]

    def encode_normalise(self):
        pass


class FeatureSelection:
    def __init__(self, sample_1, sample2):
        self.x1 = pd.DataFrame(sample_1)
        self.x2 = pd.DataFrame(sample2)
        self.x1x2_labels = list(sample_1.columns)
        self.num_items = len(self.x1x2_labels)

    def select_features(self):
        selected_labels = []

        for i in range(0, self.num_items):
            selected = input('Do you need the feature ' + str(self.x1x2_labels[i]) + '? (y/n):')

            if selected == 'y':
                selected_labels.append(str(self.x1x2_labels[i]))

        try:
            self.x2 = self.x2[selected_labels]

        except:
            self.x2 = self.x2[selected_labels[: len(selected_labels) - 1]]

        return [self.x1[selected_labels], self.x2]

    def force_dataframe_resize(self):
        reset_dataframe = pd.DataFrame()

        for sample in range(0, 2):

            if sample == 1:
                new_shape = (self.x2.shape[0], self.x1.shape[0])
            else:
                new_shape = (self.x2.shape[0], self.x1.shape[0])

            for x_feature in self.x1x2_labels:
                if sample == 1:
                    current_feature = np.array(self.x1.pop(x_feature))
                    current_feature.resize(new_shape)
                    reset_dataframe.append(pd.DataFrame(current_feature))
                    self.x1 = reset_dataframe
                    return self.x1

                else:
                    current_feature = np.array(self.x2.pop(x_feature))
                    current_feature.resize(new_shape)
                    reset_dataframe.append(pd.DataFrame(current_feature))
                    self.x2 = reset_dataframe
                    return self.x2
