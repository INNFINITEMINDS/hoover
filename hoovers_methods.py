# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:27:34 2017

implementing Adam Hoover's method 
@author: Matt
"""
import csv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

path="../../MotifCounter/"
file_name="P5_raw.csv"
#I'm very worried that this will be noisy as heck
peaks=""
iter=0

def readData(path):
    print("reading data from "+path)
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader,None)
        d = list(reader)    
    
    # import data and reshape appropriately
    data = np.array(d).astype("float") #this might be bad because it shouldn't be 0 but it is
    #just for now cut the data by a lot to make it a=managable
    X = data[:,[1,2]] #I just want the time and xyz energy I think
    

    print("done reading in the data")
    return X

def smooth(x):
    #So there are 15 obs/sec and I think the hoover paper smoothed minute by minute
    window_len=15*10
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    w=np.hanning(window_len)
    y=np.convolve(w/w.sum(),s,mode='same')
    
    print("oooohh weeeee I smoothed some data")
    return y[window_len:-window_len+1]
    


def hooverSegmentation(energy_vs_time):
    eng_df=pd.DataFrame(energy_vs_time)
    peaks=pd.DataFrame()    
    peaks["0"]=0
    t=0

#    for i in range(eng_df.shape[0]):
#        if i % 100==1:        
#            print(i)
    print(eng_df.shape[0])
    while t < eng_df.shape[0]: #forget the while loop until I can detect 1 peak correctly
#        t=i
#    for t in range(3):
        print("the t value is currently:", eng_df.iloc[t,0])
        print("the energy value is currently:", eng_df.iloc[t,1])
        thresh1=eng_df.iloc[t,1]
        thresh2=thresh1*2
        
        print("the thresholds are",thresh1,thresh2)
        
        local_t=t
        while(eng_df.iloc[local_t,1] < thresh2 and local_t < eng_df.shape[0] -1):
            local_t+=1
            if eng_df.iloc[local_t,1] < thresh1:
                #reset the thresholds
                print("reseting the threshold at index",local_t)
                thresh1=eng_df.iloc[local_t,1]
                thresh2=thresh1*2
                t=local_t
        
        #the signal exceed the second threshold or there is a min at the first data point
        
        #need some way to distinguish between the two different cases here!!        
        
        #    if thresh2 not in peaks["0"]:# and peaks.shape[0]>0:
        #        peaks=pd.concat([peaks,eng_df.loc[t]],axis=1)
        #        print("added a peak!")
        duration_t=t
        while(eng_df.iloc[duration_t,1] > thresh1 and duration_t < eng_df.shape[0] -1 ):
            duration_t+=1
        print("I think the peak is at", local_t)
        print("and it's duration is until time", duration_t)
        t=duration_t
        t+=1
            
    peaks.to_csv("peaks1.csv", mode='a')
    return peaks

def main():
    energy=readData(path+file_name)
#    plt.plot(energy[:,1])
    energy[:,1]=smooth(energy[:,1])
    
    plt.plot(energy[:,1])
#    plt.axvline(x=5, ymin=0, ymax=4.0 / max(data), linewidth=4)
    
#    peaks=hooverSegmentation(energy)
    
    print("this should be one of the peaks",max(energy[:,1]))
    return peaks
    
    
    
peaks=main()
print("yay done with main!")