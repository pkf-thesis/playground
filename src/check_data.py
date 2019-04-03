import os
import numpy as np

#data: train, test, valid
def check(data):
    print("* Checking %s data..."%data)
    file_path = "../data/msd/%s_path.txt"%data
    file = open(file_path,"r")
    lines = file.readlines()

    for line in lines:
        path =  line.strip()
        fullPath = "../sdb/data/msd/%s.npz"%path
        nparr = np.load(fullPath)['arr_0']
        sum = np.sum(nparr)
        isNaN = np.isnan(sum)
        isInf = np.isinf(sum)
        if isNaN:
            print("NaN: " + path)
        if isInf:
            print("Inf: " + path)
    file.close()

check("valid")
check("test")
check("train")