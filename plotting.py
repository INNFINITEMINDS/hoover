# -*- coding: utf-8 -*-
"""
Created on Sun May 28 22:58:45 2017

@author: Matt
"""
import csv
import matplotlib.pyplot as plt 
import numpy as np


def readData(path):
    print("reading data from "+path)
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader,None) 
        d = list(reader)    
    
    # import data and reshape appropriately
    data = np.array(d).astype("float") #this might be bad because it shouldn't be 0 but it is
    
    
    #Linear_Accel_x	Angular_Velocity_x	Linear_Accel_y	Time	
#    Angular_Velocity_z	Angular_Velocity_y	Linear_Accel_z
    
    #just for now cut the data by a lot to make it managable
    X = data[1:,1:] #we want all of them except the first col, which is indices
    
    X_df=pd.DataFrame(X,columns=["Linear_Accel_x","Angular_Velocity_x","Linear_Accel_y","Time","Angular_Velocity_z","Angular_Velocity_y","Linear_Accel_z"	])
    print("done reading in the data")
    return X_df

def makeSignalPlot(signal,title):
    plt.title(title)
    plt.plot(signal["Time"],signal["Linear_Accel_x"])
    plt.plot(signal["Time"],signal["Linear_Accel_y"])
    plt.plot(signal["Time"],signal["Linear_Accel_z"])
    plt.show()
    
if __name__ =="__main__":
    raw=readData("P12/P12_raw.csv")
    
    print(raw["Time"].iloc[np.argmax(raw["Linear_Accel_z"].iloc[:20000])])
    makeSignalPlot(raw,"raw")
    