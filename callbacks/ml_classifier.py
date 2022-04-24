"""
## some sample coding.
"""

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import pandas as pd

lb = LabelBinarizer()

array = lb.fit_transform(['malware', 'malware', 'malware', 'clean'])
print(pd.DataFrame(array))

# "True= 1, false = 0", malware = 1, clean = 0.
# Encode ispefile, isvalidsigned, label

data = pd.read_csv("model_data/test_sample.csv", delimiter=";", nrows=10)[["fileSize", "ImportFunctionCount", "PeAppendedSize"]]
print(data.head(10))
print(type(data["fileSize"]))

scaler = MinMaxScaler()
print(pd.DataFrame(scaler.fit_transform(data)))

