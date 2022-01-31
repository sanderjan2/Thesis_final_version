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
from kneed import KneeLocator

from scipy.stats import mannwhitneyu
from scipy.stats import kruskal


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


def dram(values, marker_list, type1, method, dataset, eps_pkg, eps_dram):
    idle_pkg, idle_dram = Idle_engergy()
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0.1, 1, len(values))]
    
    for i, value in enumerate(values):
        result = pd.read_csv('Results_energy/result_'+str(value)+'_'+type1+ '_' + dataset + '.csv')
        
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
        
        dram = []
        
        for label2, value2 in zip(db2.labels_, pre_dram):
            if label2 != -1:
                dram.append(value2)
        
        
        x2 = [x[0] for x in dram]
        y2 = [y[1] for y in dram]
        
        plt.scatter(x=x2, y=y2, label="K_value "+str(value), marker=marker_list[i], color=colors[i], edgecolors='black', s=75, alpha=.8)
    
    plt.xlabel("Duration in sec")
    plt.ylabel("Energy consumption in joules")
    plt.title("Dram energy consumption of " + dataset + " dataset \n with " + method)
    plt.tight_layout()
    
    plt.legend()
    plt.show()
    
    pass

def Package(values, marker_list, type1, method, dataset, eps_pkg, eps_dram):
    
    idle_pkg, idle_dram = Idle_engergy()
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0.1, 1, len(values))]
    
    for i, value in enumerate(values):
        result = pd.read_csv('Results_energy/result_'+str(value)+'_'+type1+ '_' + dataset + '.csv')
        
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
        plt.scatter(x=x1, y=y1, label="K_value "+str(value), marker=marker_list[i], color=colors[i], edgecolors='black', s=75, alpha=.8)
        # exit()

    plt.xlabel("Duration in sec")
    plt.ylabel("Energy consumption in joules")
    plt.title("Package energy consumption of " + dataset + " dataset \n with " + method)
    
    plt.legend()
    plt.tight_layout()
    
    plt.show()
    pass

def Models(values, marker_list, type1, method, dataset, eps_pkg, eps_dram):
    colors2 = [plt.cm.Spectral(each) for each in np.linspace(0.1, 1, 6)]

    models = ["Logistic_regression", "Gradient_boosting", "K_nearest"]

    for q, model in enumerate(models):
        result2 = pd.read_csv("Results_models/"+model+"_Accuracy_"+type1+"_"+ dataset + ".csv")
        baseline = pd.read_csv("Results_models/baseline_"+model+ "_" + dataset + ".csv")
        Accuracy = result2["Accuracy"].to_numpy()
        Accuracy_base = baseline["Accuracy"].to_numpy()
        
        mean_base_acc = [Accuracy_base.mean()] * len(Accuracy_base)

        plt.plot(values, Accuracy, label=model, color=colors2[q])
        plt.plot(values, mean_base_acc, label= model+"_baseline", color=colors2[q+3])
        
    plt.title("Machine learning accuracy of \n"+ method + " on the " + dataset + " dataset")

    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.ylim(.4, 1)
    plt.xlim(2, 30)
    plt.legend()
    plt.tight_layout()
    
    plt.show()



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
        # print([mean_base_acc] * len(Accuracy_base))        

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
        
        ax1.scatter(x=x1, y=y1, label="K_value "+str(value), marker=marker_list[i], color=colors[i], edgecolors='black', s=75, alpha=.8)
        ax2.scatter(x=x2, y=y2, label="K_value "+str(value), marker=marker_list[i], color=colors[i], edgecolors='black', s=75, alpha=.8)


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

# Note: this is a legacy function that shows the removal of anomalies, to make use of it replace the result part and 
# Manualy activate it in the main function.
def example(values, marker_list, type1, method, dataset, eps_pkg, eps_dram):
    
    idle_pkg, idle_dram = Idle_engergy()
    
    result = pd.read_csv('Results_energy/result_11_micro_adult.csv')
    
    result.duration *= (10**-6)
    result.pkg *= (10**-6)
    result.dram *= (10**-6)
    result.insert(0, "k_value", 11)
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
    # np_dram = np.array(pre_dram)
    db = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_pkg)
    
    pkg = []
    anomaly = []
    
    for label1, value1 in zip(db.labels_, pre_pkg):
        if label1 != -1:
            pkg.append(value1)
        else:
            anomaly.append(value1)

    
        x1 = [x[0] for x in pkg]
        y1 = [y[1] for y in pkg] 
        
        x2 = [x[0] for x in anomaly]
        y2 = [y[1] for y in anomaly]
        
            
    plt.scatter(x1, y1, color='b', edgecolors='black', s=75, alpha=.8, label="non anomalous data")
    plt.scatter(x2, y2, color='r', edgecolors='black', s=75, alpha=.8, label="anomalous data")
    
    
    plt.xlabel("Duration in sec")
    plt.ylabel("Energy consumption in joules")
    plt.title("Example of anomaly removal with package energy consumption \n of " + dataset + " dataset using " + method + " with a k value of 11")
    plt.legend()
    
    plt.show()
    
    return

# Gives the number of results that were discarded as anomalies, activate manualy from the main function
def discarded(values, marker_list, type1, method, dataset, eps_pkg, eps_dram):
    
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
        
        pkg_anomaly = []
        dram_anomaly = []
        
        for label1, label2, value1, value2 in zip(db.labels_, db2.labels_, pre_pkg, pre_dram):
            if label1 == -1:
                pkg_anomaly.append(value1)
                
            if label2 == -1:
                dram_anomaly.append(value2)
        
        
        pkg_discard = len(pkg_anomaly)
        dram_discard = len(dram_anomaly)
        
        print("---")
        print(str(pkg_discard)+" discarded for pkg k_value: "+str(value))
        print(str(dram_discard)+" discarded for dram k_value: "+str(value))
        print("---")


def man_test(values, marker_list, type1, method, dataset, eps_pkg, eps_dram, measure):
    
    idle_pkg, idle_dram = Idle_engergy()
    
    for i, value in enumerate(values):
        print("\n"+str(value)+"\n")
        # print(i)
        result = pd.read_csv('Results_energy/result_'+str(value)+'_'+type1+ '_' + dataset + '.csv')
        
        result.duration *= (10**-6)
        result.pkg *= (10**-6)
        result.dram *= (10**-6)
        
        pre_pkg1 = []
        pre_dram1 = []
        
        # np_pkg1 = np.array(pre_pkg1)
        # np_dram1 = np.array(pre_dram1)
        
        
        for dur, pkg, dram in zip(result.duration, result.pkg, result.dram):
            
            if idle_dram != None:
                pkg_no_idle = pkg - (idle_pkg * dur)
                dram_no_idle = dram - (idle_dram * dur)
            
                pre_pkg1.append((dur, pkg_no_idle))
                pre_dram1.append((dur, dram_no_idle))
            else:
                print("Warning no idle measurments where taken")
                pre_pkg1.append((dur, pkg))
                pre_dram1.append((dur, dram))
        
        np_pkg1 = np.array(pre_pkg1)
        np_dram1 = np.array(pre_dram1)
        
        db = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_pkg1)
        db2 = DBSCAN(eps=eps_dram, min_samples=5).fit(np_dram1)
        
        pkg1 = []
        dram1 = []
        
        for label1, label2, value1, value2 in zip(db.labels_, db2.labels_, pre_pkg1, pre_dram1):
            
            if label1 != -1:
                pkg1.append(value1)
            if label2 != -1:
                dram1.append(value2)
                
        x1 = [x[0] for x in pkg1]
        y1 = [y[1] for y in pkg1] 
        
        y1_avg = np.mean(y1)
        x1_avg = np.mean(x1)
        
        w1 = [x[0] for x in dram1]
        e1 = [y[1] for y in dram1]  
        
        e1_avg = np.mean(e1)      
        
        for temp2 in values:
            
            if value == temp2:
                continue
            
            result2 = pd.read_csv('Results_energy/result_'+str(temp2)+'_'+type1+ '_' + dataset + '.csv')
        
            result2.duration *= (10**-6)
            result2.pkg *= (10**-6)
            result2.dram *= (10**-6)
                
            pre_pkg2 = []
            pre_dram2 = []
            
            
            for dur, pkg, dram in zip(result2.duration, result2.pkg, result2.dram):
                
                if idle_dram != None:
                    pkg_no_idle = pkg - (idle_pkg * dur)
                    dram_no_idle = dram - (idle_dram * dur)
                
                    pre_pkg2.append((dur, pkg_no_idle))
                    pre_dram2.append((dur, dram_no_idle))
                else:
                    print("Warning no idle measurments where taken")
                    pre_pkg2.append((dur, pkg))
                    pre_dram2.append((dur, dram))
            
        
            np_pkg2 = np.array(pre_pkg2)
            np_dram2 = np.array(pre_dram2)
            
            db3 = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_pkg2)
            db4 = DBSCAN(eps=eps_dram, min_samples=5).fit(np_dram2)
            
            pkg2 = []
            dram2 = []

            for label1, label2, value1, value2 in zip(db3.labels_, db4.labels_, pre_pkg2, pre_dram2):
                if label1 != -1:
                    pkg2.append(value1)
                if label2 != -1:
                    dram2.append(value2)
            
            x2 = [x[0] for x in pkg2]
            y2 = [y[1] for y in pkg2]
            
            y2_avg = np.mean(y2)
            x2_avg = np.mean(x2)
            
            w2 = [x[0] for x in dram2]
            e2 = [y[1] for y in dram2]

            e2_avg = np.mean(e2)

            if measure == "dram":
                _ , dram_en_less = mannwhitneyu(e1, e2, alternative='less')
                _ , dram_en_greater = mannwhitneyu(e1, e2, alternative='greater')
                
                if dram_en_greater < 0.05 or dram_en_less < 0.05:
                    if e1_avg >= e2_avg:
                        print(str(temp2) + " : +")
                    else:
                        print(str(temp2) + " : -")
                else:
                    print(str(temp2) + " : O")
                
            if measure == "time":    
                _ , time_less = mannwhitneyu(x1, x2, alternative='less')
                _ , time_greater = mannwhitneyu(x1, x2, alternative='greater')
                

                if time_greater < 0.05 or time_less < 0.05:
                    if x1_avg >= x2_avg:
                        print(str(temp2) + " : +")
                    else:
                        print(str(temp2) + " : -")
                else:
                    print(str(temp2) + " : O")
        
            if measure == "pkg":
                _ , pkg_en_less = mannwhitneyu(y1, y2, alternative='less')
                _ , pkg_en_greater = mannwhitneyu(y1, y2, alternative='greater')

                if pkg_en_greater < 0.05 or pkg_en_less < 0.05:
                    if y1_avg >= y2_avg:
                        print(str(temp2) + " : +")
                    else:
                        print(str(temp2) + " : -")
                else:
                    print(str(temp2) + " : O")


# Legacy method
def take_avg(values, marker_list, type1, method, dataset, eps_pkg, eps_dram):
    idle_pkg, idle_dram = Idle_engergy()
    
    for i, value in enumerate(values):
        # print(i)
        result = pd.read_csv('Results_energy/result_'+str(value)+'_'+type1+ '_' + dataset + '.csv')
        
        result.duration *= (10**-6)
        result.pkg *= (10**-6)
        result.dram *= (10**-6)
        
        pre_pkg1 = []
        pre_dram1 = []
        
        np_pkg1 = np.array(pre_pkg1)
        np_dram1 = np.array(pre_dram1)
        
        
        for dur, pkg, dram in zip(result.duration, result.pkg, result.dram):
            
            if idle_dram != None:
                pkg_no_idle = pkg - (idle_pkg * dur)
                dram_no_idle = dram - (idle_dram * dur)
            
                pre_pkg1.append((dur, pkg_no_idle))
                pre_dram1.append((dur, dram_no_idle))
            else:
                print("Warning no idle measurments where taken")
                pre_pkg1.append((dur, pkg))
                pre_dram1.append((dur, dram))
        
        np_pkg1 = np.array(pre_pkg1)
        np_dram1 = np.array(pre_dram1)
        
        db = DBSCAN(eps=eps_pkg, min_samples=5).fit(np_pkg1)
        db2 = DBSCAN(eps=eps_dram, min_samples=5).fit(np_dram1)
        
        pkg1 = []
        dram1 = []
        
        for label1, label2, value1, value2 in zip(db.labels_, db2.labels_, pre_pkg1, pre_dram1):
            
            if label1 != -1:
                pkg1.append(value1)
            if label2 != -1:
                dram1.append(value2)
                
        # time_pkg_avg = np.mean([x[0] for x in pkg1])
        energy_pkg_avg = np.mean([y[1] for y in pkg1]) 
        
        print("")
        print(value)
        # print("pkg average time: " + str(time_pkg_avg))
        print("Pkg average energy: " + str(energy_pkg_avg))
        
        # y1_avg = np.mean(y1)
        
        time_dram_avg = np.mean([x[0] for x in dram1])
        energy_dram_avg = np.mean([y[1] for y in dram1])
        
        print("Dram average energy: " + str(energy_dram_avg))
        print("Average time: " + str(time_dram_avg))
        print("----")
    pass



if __name__ == '__main__':
    data = Load_control_file(sys.argv[1])
    
    values = data["k_values"]
    dict1, list1 = markers()
    
    misc = data["misc_results"]
    
    # take_avg(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"])
    # example(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"])
    
    if misc == "Y":
        type1 = data["misc_function"]
        
        if type1 == "man":
            man_type = data["man_function"]
            man_test(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"], man_type)
        elif type1 == "discard":
            discarded(values, list1,  data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"])
        elif type1 == "pictures":
            pictures(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"])
        elif type1 == "pkg":
            Package(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"])
        elif type1 == "dram":
            dram(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"])
        elif type1 == "models":
            Models(values, list1, data["type"], data["method"], data["data_set"], data["eps_pkg"], data["eps_dram"])
    
    