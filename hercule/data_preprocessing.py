from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas import DataFrame


class ConvertToDataframe:

    def __init__(self, data: dict):
        self.dataframe = data

    def convert_to_dataframe(self):
        return DataFrame(self.dataframe)

    def __call__(self):
        return self.convert_to_dataframe()


class FormattedData:

    def __init__(self, feature: any, data_type: str):
        self.feature = feature
        self.data_type = data_type
        self.encoder = self.get_encoder()

    def get_encoder(self):
        encoders = {
            'bool': LabelEncoder(),
            'numerical': StandardScaler()
        }
        return encoders[self.data_type]

    def __call__(self):
        result = self.encoder.fit_transform(self.feature)
        return result.T.tolist()
