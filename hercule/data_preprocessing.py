from sklearn.preprocessing import OneHotEncoder
from pandas import DataFrame

class ConverToDataframe:

    def __init__(self, data):
        self.dataframe = data
    
    def convert_to_dataframe(self):
        return DataFrame(self.dataframe)

    def __call__(self):
        return self.convert_to_dataframe()


class FormattedData:
    
    def __init__(self, feature):
        self.feature = feature

    def encode_boolean_variables(self):
        return OneHotEncoder

    def __call__(self):
        pass

from connection_engine import Session
from data_fetching import SQL

instance = Session('postgresql://api:girl11877@localhost:5432/malware')
sql = SQL(instance, 0.20, 'malware_data')
data = sql()

data_instance = ConverToDataframe(data)
print(data_instance())
