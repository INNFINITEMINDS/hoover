# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:27:34 2017

implementing Adam Hoover's method 
@author: Matt
"""
import csv
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt

path="" #../../MotifCounter/
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
    
    
    #Linear_Accel_x	Angular_Velocity_x	Linear_Accel_y	Time	
#    Angular_Velocity_z	Angular_Velocity_y	Linear_Accel_z
    
    #just for now cut the data by a lot to make it managable
    X = data[1:36000,1:] #we want all of them except the first col, which is indices
    

    print("done reading in the data")
    return X

def smooth(x): #TODO: figure out good params for the smoothing
    #So there are 15 obs/sec and I think the hoover paper smoothed minute by minute
    #not really about sigma or window size, or end behavior     
    y=x    
    
    for col in [0, 1, 2, 3, 5, 6]: # skip the time index
        y[:,col]=scipy.ndimage.filters.gaussian_filter1d(x[:,col], 50)
        #might need to have this be a per thing basis
    
    print("oooohh weeeee I smoothed some data")
    return y
    
def energyGeneration(x): 
    #energy goes first (look at eq 2 from paper)    
    window_size=1860 #they use 1 minute, so 31hz*60 that many obs per min
    
    energy=np.zeros((x.shape[0]-window_size,1)) 
    

    iter=0
    print("this thing should happen times",len(x[int(window_size/2):-int(window_size/2),0]))
    for i in x[int(window_size/2):-int(window_size/2),0]: #this is such a stupid way to write this
        
        energy[iter]=sum([abs(number) for number in x[window_size/2+iter:window_size+iter,0]])+sum([abs(number) for number in x[window_size/2+iter:window_size+iter,1]])+sum([abs(number) for number in x[window_size/2+iter:window_size+iter,2]])
        energy[iter]=1/(window_size+1) *  energy[iter]
        
        iter+=1
        
    print(iter)
    return energy
    
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
    raw=readData(path+file_name)
#    plt.plot(energy[:,1])
    smoothed=smooth(raw)
    energy=energyGeneration(smoothed) #TODO: something with the os to only do this if it doesn't exist
    
#    features=
    plt.plot(energy)
#    plt.axvline(x=5, ymin=0, ymax=4.0 / max(data), linewidth=4)
    
#    peaks=hooverSegmentation(energy)
    
    print("this should be one of the peaks",max(smoothed[:,1]))
    return peaks
    
    
    
peaks=main()
print("yay done with main!")