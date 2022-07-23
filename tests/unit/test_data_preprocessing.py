import pytest

from randforest.encoders import LabelEncoder, CategoricalEncoder
from randforest.normalizers import StandardNormalizer

mock_data = {
        'variable_x': ['person_1', 'person_2', 'person_3', 'person_4'],
        'variable_y': [True, False, False, True],
        'variable_z': [1, None, 2, None, 4, 2, 3, 10, 9]
    }

mock_data_to_fail = [1, 2, None, 'True', False]


def test_variable_data_type_fail():
    with pytest.raises(TypeError):
        data = {'variable_x': 1}
        instance = LabelEncoder(data, ['variable_x'])
        instance.check_variable_data_type()


def test_data_consistency_label_encoder_fail():
    with pytest.raises(TypeError):
        instance = LabelEncoder(mock_data, ['variable_z'])
        instance.check_data_consistency()


def test_label_encoder():
    instance = LabelEncoder(mock_data, ['variable_y'])
    assert instance.fit_transform() == {'variable_y': [0, 1, 1, 0]}


def test_data_consistency_categorical_encoder_fail():
    with pytest.raises(TypeError):
        instance = CategoricalEncoder(mock_data, ['variable_z'])
        instance.check_data_consistency()


def test_categorical_encoder():
    instance = CategoricalEncoder(mock_data, ['variable_x'])
    data = {'variable_x': ['person_1', 'person_2', 'person_3', 'person_4']}
    x = instance.build_categorization_dictionary(data['variable_x'])
    assert instance.fit_transform() == {'variable_x': [0, 1, 2, 3]}


def test_data_consistency_standard_normalizer():
    with pytest.raises(TypeError):
        data = {'variable_1': [1, 2, None, 'Hey!']}
        instance = StandardNormalizer(data, ['variable_1'])
        instance.check_data_consistency()


def test_standard_normalizer():
    data = {'variable_1': [-2, -1, 0, 1, 2]}
    instance = StandardNormalizer(data, ['variable_1'])
    print(instance.fit_transform())
    assert instance.fit_transform() == {'variable_1': [-1.414213562, -0.707106781, 0, 0.707106781, 1.414213562]}
