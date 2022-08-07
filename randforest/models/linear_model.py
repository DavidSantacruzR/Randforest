import numpy as np

from numpy.linalg import inv


class LinearRegression:

    _slots_ = [
        '_regressors_transpose',
        '_data'
    ]

    _regressors_transpose = None

    def __init__(self, regressors: dict, target: dict):
        self._regressors = np.array(list(regressors.values()))
        self._target = np.array(list(target.values()))

    def _compute_transpose_matrix(self):
        self._regressors_transpose = self._regressors.transpose()

    @staticmethod
    def _compute_matrix_multiplication(matrix_a, matrix_b):
        """
        only works with multiplication of 2 matrices
        :param matrix_a: a 2D numpy array
        :param matrix_b: a 2D numpy array
        :return: 2D numpy array
        """
        return np.matmul(matrix_a, matrix_b)

    def _predict(self):
        composed_matrix = self._compute_matrix_multiplication(self._regressors, self._regressors_transpose)
        inverse = inv(composed_matrix)
        return 0
