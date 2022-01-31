from numpy.core.defchararray import encode
import pandas as pd
import sys


missing_values = ["?", '?', "Not in universe"]

file = sys.argv[1]

# print(file)
data = pd.read_csv(file, sep=',', skipinitialspace=True, na_values=missing_values)

data.dropna(inplace=True, axis=0)
data2 = data.drop("education", axis=1)

print(data2)
data2.to_csv(file.split(".")[0]+"_clean.csv", encoding="utf-8", index=False)




# print((data.astype(str) == '?').any(1))
# print(data)