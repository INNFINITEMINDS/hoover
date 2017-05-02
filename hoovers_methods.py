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
import os

path="" #../../MotifCounter/
subj="P4" #tried P5 and I don't think there was enough variability to do this
file_name=subj+"_raw.csv"
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
    X = data[1:,1:] #we want all of them except the first col, which is indices
    

    print("done reading in the data")
    return X

def smooth(x): #TODO: figure out good params for the smoothing
    #So there are 15 obs/sec and I think the hoover paper smoothed minute by minute
    #not really about sigma or window size, or end behavior     
    y=x    
    
    for col in [0, 1, 2, 3, 5, 6]: # skip the time index
        y[:,col]=scipy.ndimage.filters.gaussian_filter1d(x[:,col], 10) #this is from the paper
        #might need to have this be a per thing basis

    smoothed_df=pd.DataFrame(y)
    smoothed_df.to_csv(subj+"_smoothed.csv")    
    
    print("oooohh weeeee I smoothed some data")
    return y
    
def energyGeneration(x): 
    #energy goes first (look at eq 2 from paper)    
    window_size=60
    #Naively try 20 seconds
    #1860 #they use 1 minute, so 31hz*60 that many obs per min
    
    energy=np.zeros((x.shape[0]-window_size,1)) 
    

    iter=0
    print("this thing should happen times",len(x[int(window_size/2):-int(window_size/2),0]))
    for i in x[int(window_size/2):-int(window_size/2),0]: #this is such a stupid way to write this
        
        energy[iter]=sum([abs(number) for number in x[window_size/2+iter:window_size+iter,0]])+sum([abs(number) for number in x[window_size/2+iter:window_size+iter,1]])+sum([abs(number) for number in x[window_size/2+iter:window_size+iter,2]])
        energy[iter]=1/(window_size+1) *  energy[iter]
        
        iter+=1
        
    energy_df=pd.DataFrame(energy)
    energy_df.to_csv(subj+"_energy.csv")
    
    
    return energy
    
def hooverSegmentation(energy): #I'm pretty sure this is working, but there isn't enough varaiblility in the energy data
    peaks=pd.DataFrame(columns=["time_of_peak","peak_value"])    

    t=0
    start_segment=0
    #Use a subset for now!
    energy=energy.loc[:]
    plt.plot(energy["Energy"])    
    plt.show()    
    
    peak_dictionary={"time":[],"value":[]}
    
    print(energy.shape[0])
    while t < energy.shape[0]: #forget the while loop until I can detect 1 peak correctly
#        print("the t value is currently:", t)
#        print("the energy value is currently:", energy["Energy"].iloc[t])
        thresh1=energy["Energy"].iloc[t]
        thresh2=thresh1*2
        
#        print("the thresholds are",thresh1,thresh2)
        
        local_t=t
        while(energy["Energy"].iloc[local_t] < thresh2 and local_t < energy.shape[0] -1):
            local_t+=1
            if energy["Energy"].iloc[local_t] < thresh1:
                #reset the thresholds
                print("reseting the threshold at index",local_t)
                thresh1=energy["Energy"].iloc[local_t]
                thresh2=thresh1*2
                
                t=local_t
        
        #the signal exceed the second threshold or you reached the end without a spike
        
        #need some way to distinguish between the two different cases here!!        
        
        duration_t=t
        while(energy["Energy"].iloc[duration_t] > thresh1 and duration_t < energy.shape[0] -1 ):
            duration_t+=1
        print("I think the peak is between", local_t)
        print("and it's duration is until time", duration_t)
        t=duration_t
              
        #something weird at index 4394        
        
        local_peak=max(energy["Energy"].iloc[start_segment:duration_t])
        
#        print(local_peak)        
#        print(energy.index.get_loc(energy["Energy"].argmax()))

#        print(energy.index.get_loc[local_peak],local_peak)
#        temp_df=pd.DataFrame({"time_of_peak":energy.index.get_loc(energy["Energy"].argmax()),"peak_value": local_peak })
        
        temp_df=pd.DataFrame(columns=["time_of_peak","peak_value"])
        hmm=energy["Energy"].iloc[start_segment:duration_t].argmax()
        print(hmm) 
        print("this should be the index",energy["Energy"].iloc[start_segment:duration_t].index.get_loc(hmm))
#        test=energy.index.get_loc(hmm)        
               
        
#        temp_df["time_of_peak"].iloc[0]=energy.index.get_loc(energy["Energy"].iloc[start_segment:duration_t].argmax())
#        temp_df["peak_value"].iloc[0]=local_peak
        peak_dictionary["time"].append(energy["Energy"].iloc[start_segment:duration_t].index.get_loc(hmm))
        peak_dictionary["value"].append(local_peak)
        
#        peaks.concat([peaks,temp_df])
        
        #maybe update the start of the buffer here?
        start_segment=t        
        
        t+=1
            
#    peaks.to_csv("peaks.csv", mode='a')
#    return peaks
        
    return peak_dictionary
    

if __name__ == "__main__":
    if not os.path.exists(path+subj+"_smoothed.csv"):
        raw=readData(path+file_name)
        smoothed=smooth(raw)
    else:
        #Linear_Accel_x	Angular_Velocity_x	Linear_Accel_y	Time	
#    Angular_Velocity_z	Angular_Velocity_y	Linear_Accel_z
        smoothed=pd.read_csv(subj+"_smoothed.csv",index_col=0, header=0,names=["Linear_Accel_x","Angular_Velocity_x","Linear_Accel_y","Time","Angular_Velocity_z","Angular_Velocity_y","Linear_Accel_z"])
    if not os.path.exists(path+subj+"_energy.csv"):
        energy=energyGeneration(smoothed) #TODO: something with the os to only do this if it doesn't exist
    else:
        energy=pd.read_csv(subj+"_energy.csv", names=["Energy"],index_col=0, header=0)
    
    
    
#    features=
#    plt.plot(energy["Energy"])

    peaks=hooverSegmentation(energy)
    

    
    
    
    print("yay done with main!")