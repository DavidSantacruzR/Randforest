from randforest.imputers import KnnImputer, MiceImputer, AverageImputer


class ImputerFactory:
    __slots__ = [
        '_method',
        '_data'
    ]

    _imputation_methods = {
        'mice': MiceImputer,
        'knn': KnnImputer,
        'average': AverageImputer
    }

    def __init__(self, data: dict, method: str):
        self._data = data
        self._method = method

    def fetch_imputation_method(self):
        return self._imputation_methods[self._method](data=self._data)
