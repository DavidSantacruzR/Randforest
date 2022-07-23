import math
from randforest.encoders import Formatter


class StandardNormalizer(Formatter):

    def check_data_consistency(self):
        for feature in self.sample:
            try:
                sum(self.sample[feature])
            except Exception as e:
                raise TypeError(f'Variable: {feature} contains non boolean data type.') from e

    @staticmethod
    def calculate_descriptive_statistics(feature):
        data_points = len(feature)
        cumulative_sum = sum(feature)
        exponential_sum = sum(number ** 2 for number in feature)
        mean_factor = cumulative_sum / data_points
        variance_factor = exponential_sum - data_points * (mean_factor ** 2)
        return mean_factor, variance_factor / data_points

    @staticmethod
    def standardize_feature(feature, mean, variance):
        return [round((data_point - mean) / math.sqrt(variance), 9) for data_point in feature]

    def fit_transform(self) -> dict:
        for feature in self.sample:
            mean, variance = self.calculate_descriptive_statistics(self.sample[feature])
            self.sample[feature] = self.standardize_feature(self.sample[feature], mean, variance)
        return self.sample
