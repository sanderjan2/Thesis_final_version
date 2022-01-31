# import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import DBSCAN

import sys
import yaml
from yaml.loader import SafeLoader
import os.path

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt


def markers():
    marker_dict = {}
    marker_list = []

    for key, value in Line2D.markers.items():
        if value != 'nothing':
            marker_dict[key] = value
            marker_list.append(key)
    
    return marker_dict, marker_list


def Idle_engergy():
    
    if not os.path.isfile("Results_energy/idle.csv"):
        print("Warning idle energy measurments missing")
        return None, None
    
    idle = pd.read_csv("Results_energy/idle.csv")
    
    idle.duration *= (10**-6)
    idle.pkg *= (10**-6)
    idle.dram *= (10**-6)
    
    for dur, pkg, dram in zip(idle.duration, idle.pkg, idle.dram):
        idle1 = pkg/dur
        idle2 = dram/dur
        
    # print(idle1, idle2)
    return idle1, idle2

def pictures(values, marker_list, type1, method, dataset, eps_pkg, eps_dram):
    
    ax1 = plt.subplot(222)
    ax2 = plt.subplot(221)
    ax3 = plt.subplot(212)

    colors = [plt.cm.Spectral(each) for each in np.linspace(0.1, 1, len(values))]

    colors2 = [plt.cm.Spectral(each) for each in np.linspace(0.1, 1, 6)]

    models = ["Logistic_regression", "Gradient_boosting", "K_nearest"]

    for q, model in enumerate(models):
        result2 = pd.read_csv("Results_models/"+model+"_Accuracy_"+type1+"_"+ dataset + ".csv")
        baseline = pd.read_csv("Results_models/baseline_"+model+ "_" + dataset + ".csv")
        Accuracy = result2["Accuracy"].to_numpy()
        Accuracy_base = baseline["Accuracy"].to_numpy()
        
        mean_base_acc = [Accuracy_base.mean()] * len(Accuracy_base)       

        ax3.plot(values, Accuracy, label=model, color=colors2[q])
        ax3.plot(values, mean_base_acc, label= model+"_baseline", color=colors2[q+3])
        
    ax3.set_title("Machine learning accuracy \n of "+ method)

    ax3.set_xlabel("K values")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(.4, 1)
    ax3.set_xlim(2, 30)
    
    idle_pkg, idle_dram = Idle_engergy()
    
    for i, value in enumerate(values):
        result = pd.read_csv('Results_energy/result_'+str(value)+'_'+type1+ '_' + dataset + '.csv')
        
        result.duration *= (10**-6)
        result.pkg *= (10**-6)
        result.dram *= (10**-6)
        result.insert(0, "k_value", value)
        total = result.pkg + result.dram
        result["total"] = total
        
        pre_pkg = []
        pre_dram = []

        for dur, pkg, dram in zip(result.duration, result.pkg, result.dram):
            
            if idle_dram != None:
                pkg_no_idle = pkg - (idle_pkg * dur)
                dram_no_idle = dram - (idle_dram * dur)
            
                pre_pkg.append((dur, pkg_no_idle))
                pre_dram.append((dur, dram_no_idle))
            else:
                print("Warning no idle measurments where taken")
                pre_pkg.append((dur, pkg))
                pre_dram.append((dur, dram))
        
        np_pkg = np.array(pre_pkg)
        np_dram = np.array(pre_dram)
        
        # Manual
        db = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_pkg)
        db2 = DBSCAN(eps=eps_dram, min_samples=5).fit(np_dram)
        
        pkg = []
        dram = []
        
        for label1, label2, value1, value2 in zip(db.labels_, db2.labels_, pre_pkg, pre_dram):
            if label1 != -1:
                pkg.append(value1)
            if label2 != -1:
                dram.append(value2)
                
        x1 = [x[0] for x in pkg]
        y1 = [y[1] for y in pkg] 
        
        x2 = [x[0] for x in dram]
        y2 = [y[1] for y in dram]
        
        # print(x1,y1)
        ax1.scatter(x=x1, y=y1, label="K_value "+str(value), marker=marker_list[i], color=colors[i], edgecolors='black', s=75, alpha=.8)
        # exit()
        ax2.scatter(x=x2, y=y2, label="K_value "+str(value), marker=marker_list[i], color=colors[i], edgecolors='black', s=75, alpha=.8)

    # result7.plot.scatter(x="duration", y="pkg", label="K_value 7", ax=ax1, marker="s", edgecolors='black', s=75, alpha=.8)
    # result7.plot.scatter(x="duration", y="dram", label="K_value 7", ax=ax2, marker="s", edgecolors='black', s=75, alpha=.8)

    ax1.set_xlabel("Duration in sec")
    ax1.set_ylabel("Energy consumption in joules")
    ax1.set_title("Package energy consumption of " + dataset + " dataset")

    ax2.set_xlabel("Duration in sec")
    ax2.set_ylabel("Energy consumption in joules")
    ax2.set_title("Dram energy consumption of " + dataset + " dataset")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()


def Load_control_file(Test_file):
    if not os.path.isfile(Test_file):
        print("Given file does not exsist")
        exit()

    if not ".yaml" in Test_file:
        print("Control-file has to be a yaml file")
        exit()

    with open(Test_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
    
    return data


if __name__ == '__main__':
    data = Load_control_file(sys.argv[1])
    values = data["k_values"]
    
    dict1, list1 = markers()
    
    pictures(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"])
    
    