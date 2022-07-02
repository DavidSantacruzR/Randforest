from hercule.data_preprocessing import FormattedData

mock_data = [True, False, False, True]


def test_one_hot_boolean_encoder():
    response = FormattedData(mock_data)
    assert response() == [1, 0, 0, 1]
