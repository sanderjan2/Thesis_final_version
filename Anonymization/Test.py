from numpy.core.numeric import NaN
import pandas as pd
import csv
from sklearn import preprocessing
import pyRAPL

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

f = open("hierarchy/capital_loss.csv", 'w')
writer = csv.writer(f)

test = pd.read_csv('adult_clean2.csv')

list1 = sorted(test.capital_loss.unique().tolist())

print(list1)
    
for element in list1:
    
    if element < 1500:
        string = [str(element), "-1500", "*"]
    elif element < 2000:
        string = [str(element), "1500-2000", "*"]
    elif element < 2250:
        string = [str(element), "2000-2250", "*"]
    elif element < 2500:
        string = [str(element), "2250-2500", "*"]
    elif element < 3000:
        string = [str(element), "2500-3000", "*"]
    elif element < 3500:
        string = [str(element), "3000-3500", "*"]    
    else:
        string = [str(element), "3500+", "*"]

    writer.writerow(string)   



# print(test.native_country.unique().tolist())

# list1 = ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands']

# for element in list1:
#     if element == "England" or element == "Germany" or element == "Portugal" or element == "France" or element == "Italy" or element == "Scotland" or element == "Greece" or element == "Ireland" or element == "Holand-Netherlands":
#         string = [str(element), "West-Europe", "Europe", "*"]
        
#     if element == "Poland" or element == "Yugoslavia" or element == "Hungary":
#         string = [str(element), "East-Europe", "Europe", "*"]
        
        
#     if element == "United-States" or element == "Canada" or element == "South" or element == "Outlying-US(Guam-USVI-etc)":
#         string = [str(element), "North-America", "America", "*"]
        
#     if element == "Cuba" or element == "Jamaica" or element == "Mexico" or element == "Honduras" or element == "Puerto-Rico" or element == "Dominican-Republic" or element == "Guatemala" :
#         string = [str(element), "Middle-America", "America", "*"]
    
#     if element == "Columbia" or element == "Cambodia" or element == "Ecuador" or element == "Haiti" or element == "El-Salvador" or element == "Peru" or element == "Trinadad&Tobago" or element == "Nicaragua":
#         string = [str(element), "South-America", "America", "*"]
    
    
#     if element == "Iran":
#         string = [str(element), "Middle-East", "North-Africa", "*"]
    
        
#     if element == "Taiwan" or element == "China" or element == "Japan" or element == "Hong":
#         string = [str(element), "North-Asia", "Asia", "*"]
        
#     if element == "India" or element == "Philippines" or element == "Thailand" or element == "Laos" or element == "Vietnam":
#         string = [str(element), "South-Asia", "Asia", "*"]
    
#     writer.writerow(string)   
        
# list1 = ['Not-in-family', 'Husband', 'Wife', 'Own-child', 
#          'Unmarried', 'Other-relative']

# for element in list1:
#     if element == "Husband" or element == "Wife":
#         string = [str(element), "Married", "*"]
#     elif element == "Own-child" or element == "Other-relative":
#         string = [str(element), "Family", "*"]
#     else:
#         string = [str(element), "Other", "*"]

#     writer.writerow(string)


# list1 = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']

# for element in list1: 
    
#     if element == "White" or element == "Black":
#         string = [str(element), "White/Black", "*"]
#     elif element == "Asian-Pac-Islander" or element == "Amer-Indian-Eskimo":
#         string = [str(element), "Non-Europian", "*"]
#     else:
#         string = [str(element), "*", "*"]
        
#     writer.writerow(string)



# list1 = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 
#     'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving', 
#     'Farming-fishing','Machine-op-inspct', 'Tech-support', 'Craft-repair', 
#     'Protective-serv', 'Armed-Forces', 'Priv-house-serv']

# for element in list1:
#     if element == 'Protective-serv' or element == 'Protective-serv':
#         string = [str(element), "Government", "*"]
#     if element == 'Other-service' or element == 'Tech-support' or element == 'Craft-repair' or element == 'Priv-house-serv' or element == 'Handlers-cleaners':
#         string = [str(element), "Services", "*"]
#     if element == 'Sales' or element == 'Adm-clerical' or element == 'Exec-managerial' or element == 'Machine-op-inspct':
#         string = [str(element), "Stores", "*"]
#     if element == 'Farming-fishing' or element == 'Prof-specialty' or element == 'Transport-moving':
#         string = [str(element), "Other", "*"]

#     writer.writerow(string)





# list1 = sorted(test.education_num.unique().tolist())

# print(test.workclass.unique())

# list1 = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 
#          'Local-gov', 'Self-emp-inc', 'Without-pay']

# for element in list1:
    
#     if "-gov" in element:
#         string = [str(element),"Government","*"]
#     elif "Self" in element:
#         string = [str(element),"Self-employed", "*"]
#     else:
#         string = [str(element),"Unknown/without", "*"]
    
    # writer.writerow(string)
    

# for i in range(1, 17):
    
#     if i == 1:
#         string = [str(i),"(1-2)","(1-3)","(1-4)","*"]
#     if i == 2:
#         string = [str(i),"(1-2)","(1-3)","(1-4)","*"]
#     if i == 3:
#         string = [str(i),"(3-4)","(2-4)","(1-4)","*"]
#     if i == 4:
#         string = [str(i),"(3-4)","(2-4)","(1-4)","*"]
    
#     if i == 5:
#         string = [str(i),"(5-6)","(5-7)","(5-8)","*"]
#     if i == 6:
#         string = [str(i),"(5-6)","(5-7)","(5-8)","*"]
#     if i == 7:
#         string = [str(i),"(7-8)","(6-8)","(5-8)","*"]
#     if i == 8:
#         string = [str(i),"(7-8)","(6-8)","(5-8)","*"]
    
#     if i == 9:
#         string = [str(i),"(9-10)","(9-11)","(9-12)","*"]
#     if i == 10:
#         string = [str(i),"(9-10)","(9-11)","(9-12)","*"]
#     if i == 11:
#         string = [str(i),"(11-12)","(10-12)","(9-12)","*"]
#     if i == 12:
#         string = [str(i),"(11-12)","(10-12)","(9-12)","*"]
    
#     if i == 13:
#         string = [str(i),"(13-14)","(13-15)","(13-16)","*"]
#     if i == 14:
#         string = [str(i),"(13-14)","(13-15)","(13-16)","*"]
#     if i == 15:
#         string = [str(i),"(15-16)","(14-16)","(13-16)","*"]
#     if i == 16:
#         string = [str(i),"(15-16)","(14-16)","(13-16)","*"]
    
#     writer.writerow(string)
# print(sorted(list1))

















# ['Age', 'workclass', 'fnlwgt', 'education', 'education-num',
    #    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    #    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    #    'class'],

# f = open("hierarchy/marital_status.csv", 'w')
# writer = csv.writer(f)

# file = pd.read_csv("Results/result2.csv")
# result2 = pd.read_csv('Results_energy/result_7_micro.csv')
# result2.duration *= (10**-6)
# result2.pkg *= (10**-6)
# result2.dram *= (10**-6)


# temp = []

# for element1, element2 in zip(result2.duration, result2.pkg):
#     # print(str(element2) + ",")
#     temp.append([element1, element2])

# test = np.array(temp)


# X = StandardScaler().fit_transform(test)

# result2.plot.scatter(x="duration", y="pkg", label="K_value 2", color='r', marker='p', edgecolors='black', s=75, alpha=.8)
# plt.show()

# print(np.reshape(result2, (-1, 1)))


# clustering = DBSCAN().fit(test)

# plt.plot(clustering)

# With thanks to https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py

# db = DBSCAN(eps=0.9, min_samples=3).fit(test)
# print(test)
# print(db.labels_)

# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)


# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = labels == k

#     xy = test[class_member_mask & core_samples_mask]
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         "o",
#         markerfacecolor=tuple(col),
#         markeredgecolor="k",
#         markersize=14,
#     )

#     xy = test[class_member_mask & ~core_samples_mask]
#     plt.plot(
#         xy[:, 0],
#         xy[:, 1],
#         "o",
#         markerfacecolor=tuple(col),
#         markeredgecolor="k",
#         markersize=6,
#     )

# plt.title("Estimated number of clusters: %d, with k=2" % n_clusters_)
# plt.show()



# print(file.fnlwgt.max())

# print(file)
# print(file.marital_status.unique())
# string = "martial-status"


# writer.writerow(["Never-married", "Unmarried", "*", "*"])
# writer.writerow(["Married-civ-spouse", "Present-married", "Married", "*"])
# writer.writerow(["Divorced","Previously-married", "Unmarried", "*"])
# writer.writerow(["Married-spouse-absent", "Absent-married","Married", "*"])
# writer.writerow(["Separated","Absent-married", "Married", "*"])
# writer.writerow(["Married-AF-spouse","Present-married","Married", "*"])
# writer.writerow(["Widowed","Previously-married","Unmarried", "*"])



# writer.writerow(["Female", "*"])

# max2 = str(file.fnlwgt.max())

# # print(max2)

# # max2 = "125489"
# max_array = []
# max_array.append(max2)

# temp2 = max2

# for i, _ in enumerate(max2):
#     temp3 = list(temp2)
#     temp3[-int(i+1)] = "*"
#     temp3 = ''.join(temp3)
#     max_array.append(temp3)
#     temp2 = temp3

# max_array.append("*")
# # print(file.columns)

# for string in file.fnlwgt.unique():
# # for string in ["12548", "125489"]:
#     string = str(string)
#     array = []
#     array.append(string)
#     string2 = string
    
#     for i , _  in enumerate(string):
#         string3 = list(string2)
#         string3[-int(i+1)] = "*"
#         string3 = ''.join(string3)
#         array.append(string3)
#         string2 = string3
    
#     array.append("*")
#     len_array = len(max_array) - len(array)
#     # print(len_array)
#     if len_array > 0:
#         for _ in range(len_array):
#             array.append('*')
#     # print(array, len(array))
    
    
    
#     writer.writerow(array)
        

# for i in range(1, 101):
#     if i < 5:
#         string = [str(i),"(0-5)","(0-10)","(0-15)","(0-20)","25-","*"]
#     elif i < 10:
#         string = [str(i),"(0-10)","(0-15)","(0-20)","25-","*","*"]
#     elif i < 15:
#         string = [str(i),"(0-15)","(0-20)","25-","*","*","*"]
#     elif i < 20:
#         string = [str(i),"(0-20)","25-","*","*","*","*"]
#     elif i < 25:
#         string = [str(i),"25-","*","*","*","*","*"]
#     elif i < 30:
#         string = [str(i),"(25-30)","(25-35)","(25-40)","(25-45)","50-","*"]
#     elif i < 35:
#         string = [str(i),"(25-35)","(25-40)","(25-45)","50-","*","*"]
#     elif i < 40:
#         string = [str(i),"(25-40)","(25-45)","50-","*","*","*"]
#     elif i < 45:
#         string = [str(i),"(40-45)","50-","*","*","*","*"]
#     elif i < 50:
#         string = [str(i),"50-","*","*","*","*","*"]
#     elif i < 55:
#         string = [str(i),"(50-55)","(50-60)","(50-65)","(50-70)","75-","*"]
#     elif i < 60:
#         string = [str(i),"(50-60)","(50-65)","(50-70)","75-","*","*"]
#     elif i < 65:
#         string = [str(i),"(50-65)","(50-70)","75-","*","*","*"]
#     elif i < 70:
#         string = [str(i),"(50-70)","75-","*","*","*","*"]
#     elif i < 75:
#         string = [str(i),"75-","*","*","*","*","*"]
#     elif i < 80:
#         string = [str(i),"(75-80)","(75-85)","(75-90)","(70-95)","100-","*"]
#     elif i < 85:
#         string = [str(i),"(75-85)","(75-90)","(75-95)","100-","*","*"]
#     elif i < 90:
#         string = [str(i),"(75-90)","(75-95)","100-","*","*","*"]
#     elif i < 95:
#         string = [str(i),"(75-95)","100-","*","*","*","*"]
#     elif i < 100:
#         string = [str(i),"100-","*","*","*","*","*"]
#     else:
#         string = [str(i),"100+","*","*","*","*","*"]
    
#     writer.writerow(string)



# for entry in file.fnlwgt.unique():
#     if (int(entry) <= 50000):
#         entry = [str(entry) ,"(0-50000)", "*"]
#         writer.writerow(entry)
#     elif (int(entry) <= 100000):
#         entry = [str(entry), "(50001-100000)", "*"]
#         writer.writerow(entry)
#     elif (int(entry) <= 150000):
#         entry = [str(entry) ,"(100001-150000)","*"]
#         writer.writerow(entry)
#     elif (int(entry) <= 200000):
#         entry = [str(entry), "(150001-200000)","*"]
#         writer.writerow(entry)
#     else:
#         entry = [str(entry), "(200000+)", "*"]
#         writer.writerow(entry)

# print(file.education.unique())