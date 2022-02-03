import pandas as pd
from sklearn.model_selection import train_test_split


class ModelBuilding:
    def __init__(self, x_sample, y_sample, target):
        self.x_sample = pd.DataFrame(x_sample)
        self.y_sample = pd.DataFrame(y_sample)

    def model_data_preparation(self):
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
