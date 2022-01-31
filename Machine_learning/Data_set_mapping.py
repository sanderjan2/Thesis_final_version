import pandas as pd
import sys
from sklearn import preprocessing

import yaml
from yaml.loader import SafeLoader


encoder = preprocessing.OrdinalEncoder()

data = data = pd.read_csv("adult_clean2.csv")
data1 = data[["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "class"]]

print(data1)
data[["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "class"]] = encoder.fit_transform( 
            data[["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "class"]])
data2 = data[["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "class"]].copy()

names = data2.columns
rename = {}

for name in names:
    rename[name] = name + "_num"

data2.rename(columns=rename, inplace=True)

print(data2.to_dict())