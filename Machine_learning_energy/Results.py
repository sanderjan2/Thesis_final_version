# import numpy as np
from email.mime import base
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
from kneed import KneeLocator

from scipy.stats import mannwhitneyu


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


def dram(values, marker_list, type1, method, dataset, eps_pkg, eps_dram, model, front):
    idle_pkg, idle_dram = Idle_engergy()
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0.1, 1, len(values)+1)]
    
    baseline = pd.read_csv("baseline_energy/baseline_energy_"+str(model)+"_"+dataset+".csv")
    baseline.duration *= (10**-6)
    baseline.pkg *= (10**-6)
    baseline.dram *= (10**-6)
    
    pre_baseline = []
    
    for dur, pkg, dram in zip(baseline.duration, baseline.pkg, baseline.dram):
        
        if idle_dram != None:
            dram_no_idle = dram - (idle_dram * dur)
        
            pre_baseline.append((dur, dram_no_idle))
        else:
            print("Warning no idle measurments where taken")
            pre_baseline.append((dur, dram))
    
    np_baseline = np.array(pre_baseline)
    
    db_baseline = DBSCAN(eps=eps_dram, min_samples=5).fit(np_baseline)
    baseline = []
    
    for label2, value2 in zip(db_baseline.labels_, pre_baseline):
        if label2 != -1 and value2[1] > 0:
            baseline.append(value2)
        
    base_x = [x[0] for x in baseline]
    base_y = [y[1] for y in baseline]
        
    plt.scatter(x=base_x, y=base_y, label="Baseline "+str(method), marker=marker_list[0], color=colors[0], edgecolors='black', s=75, alpha=.8)
    
    for i, value in enumerate(values):
        result = pd.read_csv(front+model+'/'+'result_'+str(value)+'_'+model+ '_' + dataset + '.csv')
        
        result.duration *= (10**-6)
        result.pkg *= (10**-6)
        result.dram *= (10**-6)
        result.insert(0, "k_value", value)
        total = result.pkg + result.dram
        result["total"] = total
        
        pre_dram = []

        for dur, pkg, dram in zip(result.duration, result.pkg, result.dram):
            
            if idle_dram != None:
                dram_no_idle = dram - (idle_dram * dur)
            
                pre_dram.append((dur, dram_no_idle))
            else:
                print("Warning no idle measurments where taken")
                pre_dram.append((dur, dram))
        
        np_dram = np.array(pre_dram)
        
        
        db2 = DBSCAN(eps=eps_dram, min_samples=5).fit(np_dram)
        
        # pkg = []
        dram = []
        
        for label2, value2 in zip(db2.labels_, pre_dram):
            if label2 != -1 and value2[1] > 0:
                dram.append(value2)
        
        # print("Dram " + str(len(dram)))
        
        x2 = [x[0] for x in dram]
        y2 = [y[1] for y in dram]

        plt.scatter(x=x2, y=y2, label="K_value "+str(value), marker=marker_list[i+1], color=colors[i+1], edgecolors='black', s=75, alpha=.8)

    # result7.plot.scatter(x="duration", y="pkg", label="K_value 7", ax=ax1, marker="s", edgecolors='black', s=75, alpha=.8)
    # result7.plot.scatter(x="duration", y="dram", label="K_value 7", ax=ax2, marker="s", edgecolors='black', s=75, alpha=.8)
    
    plt.xlabel("Duration in sec")
    plt.ylabel("Energy consumption in joules")
    plt.title("Dram energy consumption of " + model + " machine learning model \n on the " + dataset + " dataset")
    plt.tight_layout()
    
    plt.legend()
    plt.show()
    
    pass

def Package(values, marker_list, type1, method, dataset, eps_pkg, eps_dram, model, front):
    
    idle_pkg, idle_dram = Idle_engergy()
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0.1, 1, len(values)+1)]
    
    baseline = pd.read_csv("baseline_energy/baseline_energy_"+str(model)+"_"+dataset+".csv")
    baseline.duration *= (10**-6)
    baseline.pkg *= (10**-6)
    baseline.dram *= (10**-6)
    
    pre_baseline = []
    
    for dur, pkg, dram in zip(baseline.duration, baseline.pkg, baseline.dram):
        
        if idle_dram != None:
            pkg_no_idle = pkg - (idle_pkg * dur)
        
            pre_baseline.append((dur, pkg_no_idle))
        else:
            print("Warning no idle measurments where taken")
            pre_baseline.append((dur, dram))
    
    np_baseline = np.array(pre_baseline)
    
    # if model == "Nearest" and dataset == "":
    #     # print(eps_pkg)
    #     eps_pkg = eps_pkg /
    #     # print(eps_dram)
    
    db_baseline = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_baseline)
    baseline = []
    
    for label2, value2 in zip(db_baseline.labels_, pre_baseline):
        if label2 != -1:
            baseline.append(value2)
        
    base_x = [x[0] for x in baseline]
    base_y = [y[1] for y in baseline]
    
        
    plt.scatter(x=base_x, y=base_y, label="Baseline", marker=marker_list[0], color=colors[0], edgecolors='black', s=75, alpha=.8)
    
    for i, value in enumerate(values):
        result = pd.read_csv(front+model+'/'+'result_'+str(value)+'_'+model+ '_' + dataset + '.csv')
        
        result.duration *= (10**-6)
        result.pkg *= (10**-6)
        result.dram *= (10**-6)
        result.insert(0, "k_value", value)
        total = result.pkg + result.dram
        result["total"] = total
        
        pre_pkg = []

        for dur, pkg, dram in zip(result.duration, result.pkg, result.dram):
            
            if idle_dram != None:
                pkg_no_idle = pkg - (idle_pkg * dur)
            
                pre_pkg.append((dur, pkg_no_idle))
            else:
                print("Warning no idle measurments where taken")
                pre_pkg.append((dur, pkg))
        
        np_pkg = np.array(pre_pkg)

        # Manual
        db = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_pkg)
        
        pkg = []
        
        for label1, value1 in zip(db.labels_, pre_pkg):
            if label1 != -1:
                pkg.append(value1)
    
        # print("Package " + str(len(pkg)))
        x1 = [x[0] for x in pkg]
        y1 = [y[1] for y in pkg]
        
        # print(x1,y1)
        plt.scatter(x=x1, y=y1, label="K_value "+str(value), marker=marker_list[i+1], color=colors[i+1], edgecolors='black', s=75, alpha=.8)
        # exit()
    if model == "Nearest":
        model = "Nearest Neighbors"
    if model == "Gradient":
        model = "Gradient Boosting"
    if model == "Logistic":
        model = "Logistic Regression"

    plt.xlabel("Duration in sec")
    plt.ylabel("Energy consumption in joules")
    plt.title("Package energy consumption of the " + model + " \n machine learning model on the " + dataset + " dataset with the \n" + method + " anonymization method")
    
    plt.legend()
    plt.tight_layout()
    
    plt.show()
    pass


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


def average(values, marker_list, type1, method, dataset, eps_pkg, eps_dram, model, front):
    idle_pkg, idle_dram = Idle_engergy()
    
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0.1, 1, len(values)+1)]
    
    baseline = pd.read_csv("baseline_energy/baseline_energy_"+str(model)+"_"+dataset+".csv")
    baseline.duration *= (10**-6)
    baseline.pkg *= (10**-6)
    baseline.dram *= (10**-6)
    
    pre_baseline = []
    
    for dur, pkg, dram in zip(baseline.duration, baseline.pkg, baseline.dram):
        
        if idle_dram != None:
            pkg_no_idle = pkg - (idle_pkg * dur)
        
            pre_baseline.append((dur, pkg_no_idle))
        else:
            print("Warning no idle measurments where taken")
            pre_baseline.append((dur, dram))
    
    np_baseline = np.array(pre_baseline)
    
    # if model == "Nearest" and dataset == "":
    #     # print(eps_pkg)
    #     eps_pkg = eps_pkg /
    #     # print(eps_dram)
    
    db_baseline = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_baseline)
    baseline = []
    
    for label2, value2 in zip(db_baseline.labels_, pre_baseline):
        if label2 != -1:
            baseline.append(value2)
        
    base_x = [x[0] for x in baseline]
    base_y = [y[1] for y in baseline]
    
    print(model + " Baseline energy consumption:" +str(np.mean(base_y)))
    print(model + " Baseline execution time:" +str(np.mean(base_x))) 
    print("\n")
       

    for i, value in enumerate(values):
        result = pd.read_csv(front+model+'/'+'result_'+str(value)+'_'+model+ '_' + dataset + '.csv')
        
        result.duration *= (10**-6)
        result.pkg *= (10**-6)
        result.dram *= (10**-6)
        result.insert(0, "k_value", value)
        total = result.pkg + result.dram
        result["total"] = total
        
        pre_pkg = []

        for dur, pkg, dram in zip(result.duration, result.pkg, result.dram):
            
            if idle_dram != None:
                pkg_no_idle = pkg - (idle_pkg * dur)
            
                pre_pkg.append((dur, pkg_no_idle))
            else:
                print("Warning no idle measurments where taken")
                pre_pkg.append((dur, pkg))
        
        np_pkg = np.array(pre_pkg)

        # Manual
        db = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_pkg)
        
        pkg = []
        
        for label1, value1 in zip(db.labels_, pre_pkg):
            if label1 != -1:
                pkg.append(value1)
    
        # print("Package " + str(len(pkg)))
        x1 = [x[0] for x in pkg]
        y1 = [y[1] for y in pkg]
        
        print(model + " k-value: " + str(value) + " energy:"+ str(np.mean(y1)))
        print(model + " k-value: " + str(value) + " execution time:" +str(np.mean(x1)))
        print("\n")
        
        
        
        

def man_test(values, marker_list, type1, method, dataset, eps_pkg, eps_dram, model, front):
    
    all_values = ["baseline", 2, 3, 4, 5, 7, 9, 11, 13, 15, 18, 21, 24, 27, 30]
    # print(all_values)
    idle_pkg, idle_dram = Idle_engergy()
    
    baseline = pd.read_csv("baseline_energy/baseline_energy_"+str(model)+"_"+dataset+".csv")
    baseline.duration *= (10**-6)
    baseline.pkg *= (10**-6)
    baseline.dram *= (10**-6)
    
    pre_baseline = []
    
    for dur, pkg, dram in zip(baseline.duration, baseline.pkg, baseline.dram):
        
        if idle_pkg != None:
            pkg_no_idle = pkg - (idle_pkg * dur)
        
            pre_baseline.append((dur, pkg_no_idle))
        else:
            print("Warning no idle measurments where taken")
            pre_baseline.append((dur, pkg))
    
    np_baseline = np.array(pre_baseline)

    
    db_baseline = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_baseline)
    baseline = []
    
    for label2, value2 in zip(db_baseline.labels_, pre_baseline):
        if label2 != -1:
            baseline.append(value2)
        
    base_x = [x[0] for x in baseline]
    base_y = [y[1] for y in baseline]
    
    
    for value in all_values:
        print("\n"+str(value)+"\n")
        if value == 'baseline':
            x1 = base_x
            y1 = base_y
            x1_avg = np.mean(base_x)
            y1_avg = np.mean(base_y)
        else:
            # print(value)
            result = pd.read_csv(front+model+'/'+'result_'+str(value)+'_'+model+ '_' + dataset + '.csv')
            result.duration *= (10**-6)
            result.pkg *= (10**-6)
            
            pre_pkg1 = []
            for dur, pkg in zip(result.duration, result.pkg):
                
                if idle_dram != None:
                    pkg_no_idle = pkg - (idle_pkg * dur)
                    # dram_no_idle = dram - (idle_dram * dur)

                    pre_pkg1.append((dur, pkg_no_idle))
                    # pre_dram1.append((dur, dram_no_idle))
                else:
                    print("Warning no idle measurments where taken")
                    pre_pkg1.append((dur, pkg))

            np_pkg1 = np.array(pre_pkg1)
            db = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_pkg1)
        
            pkg1 = []
            
            for label1, value1 in zip(db.labels_, pre_pkg1):
                
                if label1 != -1:
                    pkg1.append(value1)
                    
            x1 = [x[0] for x in pkg1]
            y1 = [y[1] for y in pkg1] 
            
            y1_avg = np.mean(y1)
            x1_avg = np.mean(x1)
            
        
        for value2 in all_values:
            if value2 == value:
                print(str(value) + " : NaN")
                continue
            if value2 == 'baseline':
                x2 = base_x
                y2 = base_y
                x2_avg = np.mean(x2)
                y2_avg = np.mean(y2)
            else:
                # print(value2)
                result2 = pd.read_csv(front+model+'/'+'result_'+str(value2)+'_'+model+ '_' + dataset + '.csv')
                result2.duration *= (10**-6)
                result2.pkg *= (10**-6)
                
                # print(result2)
                
                pre_pkg2 = []
                for dur, pkg in zip(result2.duration, result2.pkg):
                    
                    if idle_pkg != None:
                        pkg_no_idle = pkg - (idle_pkg * dur)
                        # dram_no_idle = dram - (idle_dram * dur)

                        pre_pkg2.append((dur, pkg_no_idle))
                        # pre_dram1.append((dur, dram_no_idle))
                    else:
                        print("Warning no idle measurments where taken")
                        pre_pkg2.append((dur, pkg))

                np_pkg2 = np.array(pre_pkg2)
                db = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_pkg2)
            
                pkg2 = []
                
                for label1, value1 in zip(db.labels_, pre_pkg2):
                    if label1 != -1:
                        pkg2.append(value1)
                        
                x2 = [x[0] for x in pkg2]
                y2 = [y[1] for y in pkg2] 
                
                y2_avg = np.mean(y2)
                x2_avg = np.mean(x2)
                # print(x2_avg)
            _ , time_less = mannwhitneyu(x1, x2, alternative='less')
            _ , time_greater = mannwhitneyu(x1, x2, alternative='greater')
            

            if time_greater < 0.05 or time_less < 0.05:
                if x1_avg >= x2_avg:
                    print(str(value2) + " : +")
                else:
                    print(str(value2) + " : -")
            else:
                print(str(value2) + " : O")
        
          
            # _ , pkg_en_less = mannwhitneyu(y1, y2, alternative='less')
            # _ , pkg_en_greater = mannwhitneyu(y1, y2, alternative='greater')

            # if pkg_en_greater < 0.05 or pkg_en_less < 0.05:
            #     if y1_avg >= y2_avg:
            #         # pass
            #         print(str(value2) + " : +")
            #     else:
            #         print(str(value2) + " : -")
            #         # print(y1_avg, y2_avg)
            # else:
            #     print(str(value2) + " : O")
                # print(y1_avg, y2_avg)
            
                
    pass
       


if __name__ == '__main__':
    data = Load_control_file(sys.argv[1])
    values = data["k_values"]
    
    dict1, list1 = markers()
    
    # print(len(list1))
    
    models = ["Gradient"]
    # models = ["Logistic"]
    # models = ["Nearest"]
    
    for model in models:
        man_test(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"], model, data["input_file"])
        # break
        # average(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"], model, data["input_file"])
        # Package(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"], model, data["input_file"])
        # dram(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"], model, data["input_file"])
    
    # pictures(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"])
    
    