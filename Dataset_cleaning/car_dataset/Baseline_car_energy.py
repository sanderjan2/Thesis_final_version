from numpy.lib.function_base import gradient

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import sys
import numpy as np
import warnings

import yaml
from yaml.loader import SafeLoader

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
import pyRAPL


def Logistic_energy(data,target,csv):
    
    pyRAPL.setup()
    
    @pyRAPL.measureit(output=csv, number=1)
    def logistical_regression2(data, target):
        
        df_data = data.loc[:,data.columns != 'class']
        df_target = data['class']
        
        x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2)
        
        model = LogisticRegression()
        
        model.fit(x_train, y_train)
        
        predictions = model.predict(x_test)
        
        rapport = classification_report(y_test, predictions, output_dict=True)
        
        # Note, contains even more info!
        return rapport["accuracy"]
    
    csv.save()
    
    return logistical_regression2(data, target)


def logistical_regression(data,target):
    
    df_data = data.loc[:,data.columns != str(target)]
    df_target = data[str(target)]

    x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2)
   
    model = LogisticRegression(solver='lbfgs', max_iter=2500)

    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    
    rapport = classification_report(y_test, predictions, output_dict=True)
    # print(rapport)
    
    # Note, contains even more info!
    return rapport["accuracy"]


def Grad_energy(data, learning_rate, target, csv):
    
    pyRAPL.setup()
    
    @pyRAPL.measureit(output=csv, number=1)
    def Gradient_boosting2(data, lrng_rate, target):
        
        df_data = data.loc[:,data.columns != 'class']
        df_target = data['class']
        
        x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2)
        
        gradient_booster = GradientBoostingClassifier(learning_rate=lrng_rate, n_estimators=100)

        gradient_booster.fit(x_train, y_train)
        
        rapport = classification_report(y_test, gradient_booster.predict(x_test), output_dict=True)
        
        # print(rapport)
        # Note, contains even more info!
        return rapport["accuracy"]
    
    csv.save()
    accuracy = Gradient_boosting2(data, learning_rate, target)
    # Grad_boosting["k_value"][value] = value
    return accuracy


def Gradient_boosting(data, lrng_rate, target):
    
    df_data = data.loc[:,data.columns != target]
    df_target = data[target]
    
    x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2)
    
    gradient_booster = GradientBoostingClassifier(learning_rate=lrng_rate, n_estimators=100)

    gradient_booster.fit(x_train, y_train)
    
    rapport = classification_report(y_test, gradient_booster.predict(x_test), output_dict=True)
    
    # print(rapport)
    # Note, contains even more info!
    return rapport["accuracy"]
    


def Nearest_energy(data, k, target, csv):
    
    pyRAPL.setup()
    
    @pyRAPL.measureit(output=csv, number=1)
    def k_nearest2(data, k, target):
        
        df_data = data.loc[:,data.columns != 'class']
        df_target = data['class']
        
        x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2)
        
        neigh = KNeighborsClassifier(n_neighbors=k)
        
        neigh.fit(x_train, y_train)
        
        rapport = classification_report(y_test, neigh.predict(x_test), output_dict=True)
        
        # Note, contains even more info!
        # print(rapport)
        return rapport["accuracy"]

    csv.save()
    
    return k_nearest2(data, k, target)


def k_nearest(data, k, target):
    
    df_data = data.loc[:,data.columns != target]
    df_target = data[target]
    
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
    
    warnings.filterwarnings("ignore")
    model_dicts = [logistic_reg, Grad_boosting, k_near]
    
    encoder = preprocessing.OrdinalEncoder()
    
    data = pd.read_csv(input_start)

    data2 = data
    
    for column in data.columns.tolist():
        data2[[column]] = encoder.fit_transform(data[[column]])
        data2[column] = data2[column].astype(float)

    data = data2
    

    csv_output_log = pyRAPL.outputs.CSVOutput('baseline_energy/baseline_energy_Logistic_'+ dataset +' .csv')
    for _ in range(50):
        accuracy_log = Logistic_energy(data, target, csv_output_log)
    logistic_acc_en = logistical_regression(data, target)
    
    csv_output_Grad = pyRAPL.outputs.CSVOutput('baseline_energy/baseline_energy_Gradien_'+ dataset +' .csv')
    for _ in range(50):
        accuracy_Grad = Grad_energy(data, learning_rate, target, csv_output_Grad)
    Grad_acc_en = accuracy_Grad
    
    csv_output_Nearest = pyRAPL.outputs.CSVOutput('baseline_energy/baseline_energy_Nearest_'+ dataset +' .csv')
    for _ in range(50):
        accuracy_Nearest = Nearest_energy(data, nearest_neighbors, target, csv_output_Nearest)
    Nearest_acc_en = accuracy_Nearest
    
    # df = pd.DataFrame.from_dict(Grad_boosting).sort_index()
    # df.to_csv("baseline/baseline.csv", index=False, header=True)
    # for i, model in enumerate(model_dicts):
        # df = pd.DataFrame.from_dict(model).sort_index()
        # df.to_csv("baseline/baseline_"+models[i]+"_"+ dataset +".csv", index=False, header=True)