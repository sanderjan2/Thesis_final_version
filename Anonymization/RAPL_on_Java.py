import pyRAPL
import subprocess
import os.path
from subprocess import STDOUT, PIPE
import sys
import yaml
from yaml.loader import SafeLoader
import time

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

def create_gen(data):
    f = open("hierarchy.txt", "w")
    for key, value in data["hierarchy"].items():
        f.write(key + "," + value + "," + "\n")
    f.close()

def create_aggro(data):
    # print("hi")
    f = open("hierarchy.txt", "w")
    for key, value in data["hierarchy"].items():
        f.write(key + "," + value + "," + data["types"][key] + "," + data["micro_function"][key] + "\n")
    f.close()

def create_hierachy(data):
    if data["type"] == "micro":
        create_aggro(data)
    elif data["type"] == "general":
        create_gen(data)
    else:
        print("Invalid type given")
        exit()

def measurment(k, itterations, input_file, class_name, suppression, data_type, dataset): 
    pyRAPL.setup()
    
    if os.path.isfile('Results/result_'+str(k)+'_'+data_type+'.csv'):
        os.remove('Results/result_'+str(k)+'_'+data_type+'.csv')

    csv_output = pyRAPL.outputs.CSVOutput('Results_energy/result_'+str(k)+'_'+data_type+ '_'+ dataset +'.csv')

    @pyRAPL.measureit(output=csv_output, number=1)
    def Energy_consumption():

        subprocess.run(['sudo', 'java', '-cp', '.:libraries/*', class_name, 
                        str(k), input_file, str(suppression), dataset], capture_output=False)
        
    for _ in range(itterations):
        Energy_consumption()
    
    csv_output.save()
        
# activate once manualy to take idle energy measurments (it takes the average of 30 runs)
def idle_measurments():
    
    if os.path.isfile("Results/idle.csv"):
        os.remove("Results/idle.csv")
        
    pyRAPL.setup()

    csv_output2 = pyRAPL.outputs.CSVOutput('Results_energy/idle.csv')
    
    @pyRAPL.measureit(output=csv_output2, number=30)
    def wait():
        time.sleep(1)

    wait()
    csv_output2.save()
    
    
if __name__ == '__main__':
    
    Test_file = sys.argv[1]
    
    data = Load_control_file(Test_file)
    
    create_hierachy(data)
    
    # exit()
    
    for k in data["k_values"]:
        measurment(k, int(data["iterations"]), data["input_file"], data["class_name"], 
                        data["suppression_limit"], data["type"], data["data_set"])
    
    # idle_measurments()
    