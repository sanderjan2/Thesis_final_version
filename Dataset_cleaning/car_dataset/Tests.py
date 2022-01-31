import pandas as pd
import sys

import pandas as pd
import csv

file = sys.argv[1]

# print(file)
data = pd.read_csv(file, sep=',', skipinitialspace=False)

f = open("hierarchies/safety.csv", 'w')
writer = csv.writer(f)

items = ["low", "med", "high"]


for item in items:
    if item == "low":
        string = [item, "*"]
        
    if item == "med":
        string = [item, "*"]
        
    if item == "high":
        string = [item, "*"]
    
    writer.writerow(string)


# for item in items:
#     if item == "vhigh":
#         string = [item, "vhigh/high", "*"]

#     if item == "high":
#         string = [item, "vhigh/high", "*"]

#     if item == "med":
#         string = [item, "low/med", "*"]

#     if item == "low":
#         string = [item, "low/med", "*"]

#     writer.writerow(string)

# list1 = sorted(test.capital_loss.unique().tolist())

# print(list1)
    
# for element in list1:
    
#     if element < 1500:
#         string = [str(element), "-1500", "*"]
#     elif element < 2000:
#         string = [str(element), "1500-2000", "*"]
#     elif element < 2250:
#         string = [str(element), "2000-2250", "*"]
#     elif element < 2500:
#         string = [str(element), "2250-2500", "*"]
#     elif element < 3000:
#         string = [str(element), "2500-3000", "*"]
#     elif element < 3500:
#         string = [str(element), "3000-3500", "*"]    
#     else:
#         string = [str(element), "3500+", "*"]

#     writer.writerow(string)   

# print(data)