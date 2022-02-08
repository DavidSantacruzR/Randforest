import unittest
import random
from mlClassifier.dependencies.logistic_model_preprocessing import ModelBuilding
from pandas import DataFrame


def create_random_sample(n_features: int):

    raw_sample = {}

    for feature in range(1, n_features):
        name = "feature_" + str(feature)
        raw_sample[name] = random.sample(range(1000), k=100)

    return raw_sample


class TestDataSplit(unittest.TestCase):

    model = ModelBuilding(DataFrame(create_random_sample(3)), DataFrame(create_random_sample(3)))
    samples = {}

    def test_number_of_sub_samples(self):

        [x_train, x_test, y_train, y_test] = self.model.model_data_preparation()
        self.samples['x_train'] = x_train
        self.samples['x_test'] = x_test
        self.samples['y_train'] = y_train
        self.samples['y_test'] = y_test

        number_of_sub_samples = len(self.samples)
        self.assertCountEqual(str(number_of_sub_samples), "4", "unequal sample size")


if __name__ == '__main__':
    unittest.main()
