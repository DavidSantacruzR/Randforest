

class Encoders:

    _encoded = {}

    def __init__(self, dataset: dict, selected_features: list) -> None:
        self.dataset = dataset
        self.selected = selected_features
        self.to_encode = self.extract_features()
        if not isinstance(dataset, dict):
            raise ValueError('Dataset must be a dictionary.')

    def extract_features(self) -> set:
        """
        Extracts a set of features to be transformed according to the selected
        encoder implementation.
        :return:
        """
        return {self.dataset[feature] for feature in self.selected}

    def factory_method(self) -> None:
        pass

    def _check_variable_data_type(self):
        for feature in self.to_encode:
            if not isinstance(feature, list):
                raise ValueError(f'Variable: {feature} must be a list.')


class LabelEncoder(Encoders):

    """
    Basic label encoder replacing every item in each feature for either 0 or 1 depending on variable is
    True or False.
    """

    def check_variable_data_type(self):
        for feature in self.to_encode:
            if not isinstance(feature, list):
                raise ValueError(f'Variable: {feature} must be a list.')

    def check_data_consistency(self):
        for feature in self.to_encode:
            if len(set(feature)) != 2:
                raise ValueError(f'Variable: {feature} contains non boolean data type.')

    def fit_transform(self) -> dict:
        for feature in self.to_encode:
            self._encoded[feature] = [0 if value is True else False for value in feature]
        return self._encoded


class CategoricalEncoder(Encoders):

    """
    Categorical encoder that replaces every unique element in feature for it's numerical representation.
    """

    def check_data_consistency(self):
        for feature in self.to_encode:
            if None in feature:
                raise ValueError(f'None type cannot be categorized, resolve it in Variable: {feature}.')

    @staticmethod
    def build_categorization_dictionary(feature):
        for enumeration, item in enumerate(set(feature)):
            return {item: enumeration}

    def fit_transform(self):
        for feature in self.to_encode:
            categorization = self.build_categorization_dictionary(feature)
            self._encoded[feature] = [categorization[value] for value in feature]


class Normalizers:

    def __init__(self, dataset: dict):
        self.dataset = dataset

    def factory_method(self):
        pass
