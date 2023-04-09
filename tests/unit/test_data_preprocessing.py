import pytest

from randforest.encoders import LabelEncoder, CategoricalEncoder
from randforest.normalizers import StandardNormalizer
from randforest.factories.imputers import ImputationFactory

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
    assert instance.fit_transform() == {'variable_x': [0, 1, 2, 3]}


def test_data_consistency_standard_normalizer():
    with pytest.raises(TypeError):
        data = {'variable_1': [1, 2, None, 'Hey!']}
        instance = StandardNormalizer(data, ['variable_1'])
        instance.check_data_consistency()


def test_standard_normalizer():
    data = {'variable_1': [-2, -1, 0, 1, 2]}
    instance = StandardNormalizer(data, ['variable_1'])
    assert instance.fit_transform() == {'variable_1': [-1.414213562, -0.707106781, 0, 0.707106781, 1.414213562]}


def test_mice_data_imputation():
    """
    Credit worthiness is the target variable for the test.
    """
    imputation_mock_data = {
        'age': [35, 20, 56, 47, 38],
        'wages': [100, 200, 150, 320, 80],
        'dependents': [1, 2, 0, 1, 1],
        'experience': [8, 8, 5, 3, 9],
        'credit_worthy': [0.01, 0.012, 0.02, 0.019, 0.3]
    }

    to_impute_data = {
        'age': [35, 20, 56, None, 38],
        'wages': [100, 200, None, 320, 80],
        'dependents': [None, 2, 0, 1, 1],
        'experience': [8, 8, 5, None, 9],
        'credit_worthy': [0.01, 0.012, 0.02, 0.019, None]
    }

    instance = ImputationFactory(to_impute_data, method='mice', target='credit_worthy').fetch_imputation_method()
    print(instance._create_missing_values_index_matrix(), instance._index_matrix)
    print(instance._get_base_matrix(), instance._base_matrix)
    assert instance.calculate_result_matrix() == imputation_mock_data
