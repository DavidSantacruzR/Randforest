import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_preprocessing import FeatureSelection


class ModelBuilding:
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

        if self.x_sample.shape[0] > self.y_sample.shape[0]:
            needed_reshape = 'x_sample'
        else:
            needed_reshape = 'y_sample'

        # When resizing is required. USe the custom built function.

        if needed_reshape == 'x_sample':
            self.x_sample = FeatureSelection(self.x_sample, self.y_sample).force_dataframe_resize()
        elif needed_reshape == 'y_sample':
            self.y_sample = FeatureSelection(self.y_sample, self.x_sample).force_dataframe_resize()

    def data_preparation(self):
        print(self.x_sample.head(10))
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_sample,
            self.y_sample,
            test_size=0.33,
            random_state=42)

        return [x_train, x_test, y_train, y_test]


class PredictionService:
    def __init__(self):
        pass
