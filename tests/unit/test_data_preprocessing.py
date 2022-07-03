import numpy as np
from hercule.data_preprocessing import FormattedData

mock_data = [True, False, False, True]
numerical_data = [1, 50, 4, -2, 5, 0]
mean = sum(numerical_data) / len(numerical_data)
squared_numbers = [value ** 2 for value in numerical_data]
variance = sum(squared_numbers) / len(numerical_data) - mean ** 2
deviation = variance ** (1/2)


def standard_normaliser(data):
    return [[(item - mean) / deviation for item in data]]


def test_encode_boolean_variables():
    response = FormattedData(mock_data, 'bool')

    assert response() == [1, 0, 0, 1]


def test_normalise_numerical_variables():
    response = FormattedData(np.array([numerical_data]).T, 'numerical')
    expected_result = standard_normaliser(numerical_data)
    print(expected_result)
    assert response() == expected_result
