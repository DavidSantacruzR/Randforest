from hercule.data_preprocessing import FormattedData


mock_data = [True, False, False, True]

def test_one_hot_boolean_encoder():
    response = []
    assert response == [0, 1, 1, 0]
