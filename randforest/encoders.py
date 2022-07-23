import abc


class Formatter(abc.ABC):

    def __init__(self, dataset: dict, selected_features: list) -> None:
        self.dataset = dataset
        self.selected = selected_features
        self.sample = self.extract_features()
        if not isinstance(dataset, dict):
            raise TypeError('Dataset must be a dictionary.')
        self.check_variable_data_type()
        self.check_data_consistency()

    def extract_features(self) -> dict:
        """
        Extracts a set of features to be transformed according to the selected
        encoder implementation.
        """
        return {feature: self.dataset[feature] for feature in self.selected}

    def check_variable_data_type(self):
        for feature in self.sample:
            if not isinstance(self.sample[feature], list):
                raise TypeError(f'Variable: {feature} must be a list.')

    @abc.abstractmethod
    def check_data_consistency(self):
        pass

    @abc.abstractmethod
    def fit_transform(self):
        pass


class LabelEncoder(Formatter):
    """
    Basic label encoder replacing every item in each feature for either 0 or 1 depending on variable is
    True or False.
    """

    def check_data_consistency(self):
        for feature in self.sample:
            if len(set(self.sample[feature])) != 2:
                raise TypeError(f'Variable: {feature} contains non boolean data type.')

    def fit_transform(self) -> dict:
        for feature in self.sample:
            self.sample[feature] = [0 if value is True else 1 for value in self.sample[feature]]
        return self.sample


class CategoricalEncoder(Formatter):
    """
    Categorical encoder that replaces every unique element in feature for it's numerical representation.
    """

    def check_data_consistency(self):
        for feature in self.sample:
            if None in self.sample[feature]:
                raise TypeError(f'None type cannot be categorized, resolve it in Variable: {feature}.')

    @staticmethod
    def build_categorization_dictionary(feature):
        unique_values = list(dict.fromkeys(feature))
        return {item: enumeration for enumeration, item in enumerate(unique_values)}

    def fit_transform(self):
        for feature in self.sample:
            categorization = self.build_categorization_dictionary(self.sample[feature])
            self.sample[feature] = [categorization[value] for value in self.sample[feature]]
        return self.sample
