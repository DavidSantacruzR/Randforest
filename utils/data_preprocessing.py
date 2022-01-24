import pandas as pd


class DataPreProcessing:
    def __init__(self, train_sample_name, test_sample_name):
        self.train = train_sample_name
        self.test = test_sample_name

    def import_data(self):
        self.train = pd.DataFrame(pd.read_csv(self.train, delimiter=';'))
        self.test = pd.DataFrame(pd.read_csv(self.test, delimiter=';'))

    def remove_null_values(self):
        self.train = self.train.dropna()
        self.test = self.test.dropna()

        return [self.train, self.test]

    def resize_dataframe(self):
        """
        ## use the np.array resize and the DataFrame.append or concatenate method.
        ## Iteratively add each resized feature to the new dataframe.
        :return:
        """

        if self.train.shape[0] > self.test.shape[0]:
            rows_to_drop = int(self.train.shape[0] - self.test.shape[0])
            self.train = self.train.iloc[rows_to_drop:, ]

            return self.train, self.test

        else:
            rows_to_drop = int(self.train.shape[0] - self.test.shape[0])
            self.test = self.test.iloc[rows_to_drop:, ]

            return self.test, self.test


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

            self.x2 = self.x2[selected_labels[: len(selected_labels)-1]]

        return [self.x1[selected_labels], self.x2]
