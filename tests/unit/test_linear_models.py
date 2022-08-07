from randforest.models import LinearRegression


def test_linear_regression_estimator():
    regressors = {
        'variable_1': [20, 30, 22, 18, 27],
        'variable_2': [200, 150, 120, 310, 280],
        'variable_3': [3, 3, 4, 5, 7]
    }

    target = {
        'target': [1, 1, 0, 0, 1]
    }

    to_predict = {
        'variable_1': [19, 24, 33],
        'variable_2': [180, 110, 204],
        'variable_3': [4, 6, 9]
    }

    instance = LinearRegression(regressors, target, to_predict)
    assert instance._transform_estimators_vector() == [
        0.09366783409371948,
        0.00403455168376235,
        -0.17170184536614608,
        -1.691664155139609
    ]
