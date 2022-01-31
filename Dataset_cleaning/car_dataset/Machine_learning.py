from numpy.lib.function_base import gradient

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import sys
import numpy as np

import yaml
from yaml.loader import SafeLoader

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split


def logistical_regression(data, target):
    
    df_data = data.loc[:,data.columns != 'class']
    df_target = data['class']
    
    x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2)
    
    model = LogisticRegression()
    
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    
    rapport = classification_report(y_test, predictions, output_dict=True)
    
    # Note, contains even more info!
    return rapport["accuracy"]


def Gradient_boosting(data, lrng_rate, target):
    
    df_data = data.loc[:,data.columns != 'class']
    df_target = data['class']
    
    x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2)
    
    gradient_booster = GradientBoostingClassifier(learning_rate=lrng_rate, n_estimators=100)

    gradient_booster.fit(x_train, y_train)
    
    rapport = classification_report(y_test, gradient_booster.predict(x_test), output_dict=True)
    
    # print(rapport)
    # Note, contains even more info!
    return rapport["accuracy"]
    


def k_nearest(data, k, target):
    
    df_data = data.loc[:,data.columns != 'class']
    df_target = data['class']
    
    x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2)
    
    neigh = KNeighborsClassifier(n_neighbors=k)
    
    neigh.fit(x_train, y_train)
    
    rapport = classification_report(y_test, neigh.predict(x_test), output_dict=True)
    
    # Note, contains even more info!
    # print(rapport)
    return rapport["accuracy"]


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        control = yaml.load(f, SafeLoader)
    
    k_values = control["k_values"]
    input_start = control["input_start"]
    
    learning_rate = control["learning_rate"]
    nearest_neighbors = control["nearest_neighbors"]
    
    models = ["Logistic_regression", "Gradient_boosting", "K_nearest"]
    
    target = control["target"]
    dataset = control["dataset"]
    
    logistic_reg = {}
    logistic_reg["Accuracy"] = {}
    logistic_reg["k_value"] = {}
    
    Grad_boosting = {}
    Grad_boosting["Accuracy"] = {}
    Grad_boosting["k_value"] = {}
    
    k_near = {}
    k_near["Accuracy"] = {}
    k_near["k_value"] = {}
    
    model_dicts = [logistic_reg, Grad_boosting, k_near]
    
    # data = pd.read_csv("Results_anonymisation/Micro_Anony_2.csv")
    
    # Gradient_boosting(data, learning_rate)
    # k_nearest(data, nearest_neighbors)
    
    for value in k_values:
        data = pd.read_csv(input_start + str(value) + ".csv")
        
        logistic_reg["Accuracy"][value] = logistical_regression(data, target)
        logistic_reg["k_value"][value] = value
        
        Grad_boosting["Accuracy"][value] = Gradient_boosting(data, learning_rate, target)
        Grad_boosting["k_value"][value] = value
        
        k_near["Accuracy"][value] = k_nearest(data, nearest_neighbors, target)
        k_near["k_value"][value] = value
    
    for i, model in enumerate(model_dicts):
        df = pd.DataFrame.from_dict(model).sort_index()
        df.to_csv("Results_Machine_models/"+models[i]+"_Accuracy_"+control["method"]+ "_" + dataset +".csv", index=False, header=True)
    
    
    