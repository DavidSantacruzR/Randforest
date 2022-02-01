from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas import DataFrame, read_csv
from numpy import array

data = DataFrame(read_csv("train_sample.csv", delimiter=";", nrows=10)).dropna()

data.to_csv("before_encoder_normalise.csv")  # data before processing.
labels = list(data.columns)
to_encode = {}

# Need to encode variables which are text.
for element in labels:
    data_type = str(data[element].dtype)
    to_encode[element] = data_type  # adds new element to dictionary.

    if to_encode[element] == 'int64':
        print('scaler running', element)
        scaler = StandardScaler()
        reshape_data = array(data[element]).reshape(-1, 1)
        data[element] = scaler.fit_transform(reshape_data)

    elif to_encode[element] == 'float64':
        print('scaler running', element, to_encode[element] == 'float64')
        scaler = StandardScaler()
        reshape_data = array(data[element]).reshape(-1, 1)
        data[element] = scaler.fit_transform(reshape_data)

    else:
        print('encoder running', element, to_encode[element] != 'int64')
        label_encoder = LabelEncoder()
        data[element] = label_encoder.fit_transform(data[element])

print(to_encode)
data.to_csv("after_encoder_normalise.csv")  # data after processing.

