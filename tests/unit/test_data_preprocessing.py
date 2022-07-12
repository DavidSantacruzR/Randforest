import pytest

from hercule.encoders import LabelEncoder, CategoricalEncoder

mock_data = {
        'variable_x': ['person_1', 'person_2', 'person_3', 'person_4'],
        'variable_y': [True, False, False, True],
        'variable_z': [1, None, 2, None, 4, 2, 3, 10, 9]
    }

mock_data_to_fail = [1, 2, None, 'True', False]


def test_variable_data_type_correct():
    try:
        instance = LabelEncoder(mock_data, ['variable_y'])
        instance.check_variable_data_type()
    except TypeError as e:
        assert False, f'check_variable_data_type raised an exception {e}'


def test_variable_data_type_fail():
    with pytest.raises(TypeError):
        data = {'variable_x': 1}
        instance = LabelEncoder(data, ['variable_x'])
        instance.check_variable_data_type()


def test_data_consistency_label_encoder():
    try:
        instance = LabelEncoder(mock_data, ['variable_y'])
        instance.check_data_consistency()
    except ValueError as e:
        assert False, f'check_data_consistency raised an exception {e}'


def test_data_consistency_label_encoder_fail():
    with pytest.raises(ValueError):
        instance = LabelEncoder(mock_data, ['variable_z'])
        instance.check_data_consistency()


def test_label_encoder():
    instance = LabelEncoder(mock_data, ['variable_y'])
    assert instance.fit_transform() == {'variable_y': [0, 1, 1, 0]}


def test_data_consistency_categorical_encoder():
    try:
        instance = CategoricalEncoder(mock_data, ['variable_x'])
        instance.check_data_consistency()
    except ValueError as e:
        assert False, f'check_data_consistency raised an exception {e}'


def test_data_consistency_categorical_encoder_fail():
    with pytest.raises(ValueError):
        instance = CategoricalEncoder(mock_data, ['variable_z'])
        instance.check_data_consistency()


def test_categorical_encoder():
    instance = CategoricalEncoder(mock_data, ['variable_x'])
    data = {'variable_x': ['person_1', 'person_2', 'person_3', 'person_4']}
    x = instance.build_categorization_dictionary(data['variable_x'])
    assert instance.fit_transform() == {'variable_x': [0, 1, 2, 3]}
