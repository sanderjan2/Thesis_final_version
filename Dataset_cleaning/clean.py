from numpy.core.defchararray import encode
import pandas as pd
import sys
from sklearn import preprocessing


missing_values = ["?", '?', "Not in universe"]

file = sys.argv[1]

# print(file)
data = pd.read_csv(file, sep=',', skipinitialspace=True, na_values=missing_values)

# data2 = data.dropna(inplace=True, axis=0)
# 
encoder = preprocessing.OrdinalEncoder()
# "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "class"

data[["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "class"]] = encoder.fit_transform(
    data[["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "class"]]
)

names = data.columns
rename = {}

for name in names:
    if "-" in name:
        rename[name] = name.replace("-","_")
        
data.rename(columns=rename, inplace=True)

print(data)

data.to_csv(file.split(".")[0]+"_micro.csv", encoding="utf-8", index=False)



