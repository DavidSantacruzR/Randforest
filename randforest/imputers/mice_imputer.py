

class MiceImputer:
    """
    The most important assumption in the mice algorithm
    is that data is missing completely at random.
    """
    _base_matrix = {}
    _index_matrix = {}
    _comparison_matrix = {}
    _regressed_feature = {}
    _regressors = {}
    _regression_parameters = {}

    def __init__(self, data: dict, target: str, convergence=False):
        self._data = data
        self._convergence = convergence
        self._target = target

    @staticmethod
    def _get_feature_average(feature) -> float:
        cumulative_sum = sum(point if point is not None else 0 for point in feature)
        n_values = sum(point is not None for point in feature)
        null_values = len(feature) - n_values
        return cumulative_sum / (n_values - null_values)

    @staticmethod
    def _fill_missing_values(feature, average: float) -> list:
        for index, value in enumerate(feature):
            if value is None:
                feature[index] = average
        return feature

    def _create_missing_values_index_matrix(self):
        for feature in self._data:
            self._index_matrix[feature] = [
                0 if value is None else 1 for value in self._data[feature]
            ]

    def _update_missing_values_index_matrix(self):
        pass

    def _get_base_matrix(self):
        for feature in self._data:
            if feature != self._target:
                avg = self._get_feature_average(self._data[feature])
                values = self._fill_missing_values(self._data[feature], avg)
                self._base_matrix[feature] = values

    def _calculate_differential_matrix(self):
        self._get_base_matrix()
        return self._base_matrix

    def _update_regression_parameters(self, target, regressors):
        pass

    def _insert_estimated_value(self):
        pass

    @staticmethod
    def calculate_result_matrix():
        return 0
