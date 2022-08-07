import numpy as np

from numpy.linalg import inv


class LinearRegression:

    _slots_ = [
        '_regressors_transpose',
        '_data'
    ]

    _regressors_transpose = None
    _target_transpose = None

    def __init__(self, regressors: dict, target: dict, to_predict: dict):
        self._features = list(regressors.values())
        self._regressors = None
        self._target = np.array(list(target.values()))
        self._predict = to_predict
        self._completed_target = self._target

    def _compute_transpose_matrix(self):
        self._regressors_transpose = self._regressors.transpose()
        self._target_transpose = self._target.transpose()

    def _append_intercept_to_regressors(self):
        intercept_elements = len(self._features[0])
        self._features.append([1] * intercept_elements)
        self._regressors = np.array(self._features)

    @staticmethod
    def _compute_matrix_multiplication(matrix_a, matrix_b):
        """
        only works with multiplication of 2 matrices
        :param matrix_a: a 2D numpy array
        :param matrix_b: a 2D numpy array
        :return: 2D numpy array
        """
        return np.matmul(matrix_a, matrix_b)

    def _estimators_vector(self):
        self._append_intercept_to_regressors()
        self._compute_transpose_matrix()
        composed_matrix = self._compute_matrix_multiplication(self._regressors, self._regressors_transpose)
        inverse = inv(composed_matrix)
        result_vector = self._compute_matrix_multiplication(self._regressors, self._target_transpose)
        return self._compute_matrix_multiplication(inverse, result_vector)

    def _transform_estimators_vector(self):
        estimators = list(self._estimators_vector())
        return [estimator[0] for estimator in estimators]

    def _predictions_vector(self):
        for predicted_value in self._predict:
            return 0
