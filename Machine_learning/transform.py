from numpy.core.defchararray import encode
import pandas as pd
import sys
from sklearn import preprocessing

import yaml
from yaml.loader import SafeLoader


def clean(file):
    missing_values = ["?", '?', "Not in universe"]
    # print(file)
    data = pd.read_csv(file, sep=',', skipinitialspace=True, na_values=missing_values)

    # data2 = data.dropna(inplace=True, axis=0)
    # 
    # encoder = preprocessing.OrdinalEncoder()
    # # "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "class"

    # data[["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "class"]] = encoder.fit_transform(
    #     data[["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "class"]]
    # )

    names = data.columns
    rename = {}

    for name in names:
        if "-" in name:
            rename[name] = name.replace("-","_")
            
    data.rename(columns=rename, inplace=True)

# print(data)

# data.to_csv(file.split(".")[0]+"_micro.csv", encoding="utf-8", index=False)


def transform(data, all):
    encoder = preprocessing.OrdinalEncoder()
    
    if all:
        print(data)
        data [["Age","workclass","fnlwgt","education_num","marital_status","occupation","relationship","race","sex","capital_gain",
               "capital_loss","hours_per_week","native_country","class"]] = encoder.fit_transform(data [["Age","workclass","fnlwgt","education_num","marital_status",
                "occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","class"]])
        print(data)
    
    else:
        data[["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "class"]] = encoder.fit_transform( 
            data[["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "class"]])

    return data

# Only for non continues value fields.
def new_transform(data):
    data2 = data
    encoder = preprocessing.OrdinalEncoder()
    
    for column in data.columns.tolist():
        data2[[column]] = encoder.fit_transform(data[[column]])
        data2[column] = data2[column].astype(float)
    
    return data2


if __name__ == '__main__':
    file = sys.argv[1]
    
    with open(file) as f:
        control = yaml.load(f, SafeLoader)
    
    k_values = control["k_values"]
    
    if control["transform"]:
        
        for value in k_values:
            
            data = pd.read_csv(control["pre_input"] + str(value) + "_"+ control["dataset"] + ".csv")
            # print(data)
            data = new_transform(data)
            # if control["method"] == "general":
            #     data = transform(data, True)
            # else:
            #     data = transform(data, False)
                
            data.to_csv(control["input_start"]+str(value)+ "_" + control["dataset"]+ ".csv", encoding="utf-8", index=False)
            
        
    