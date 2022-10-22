from randforest.imputers import KnnImputer, MiceImputer, AverageImputer


class ImputationFactory:
    __slots__ = [
        '_method',
        '_data',
        '_kwargs'
    ]

    _imputation_methods = {
        'mice': MiceImputer,
        'knn': KnnImputer,
        'average': AverageImputer
    }

    def __init__(self, data: dict, method: str, **kwargs):
        self._data = data
        self._method = method
        self._kwargs = dict(**kwargs)

    def fetch_imputation_method(self):
        convergence = self._kwargs.get('convergence')
        target = self._kwargs.get('target')
        return self._imputation_methods[self._method](
            data=self._data,
            convergence=convergence,
            target=target
        )
