# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:27:34 2017

implementing Adam Hoover's method 
@author: Matt
"""
import csv
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB

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

def gauss(t,n,sigma): #In the paper n=31 sigma=10
    num=np.exp(- t**2 / (2*sigma**2))
    print(num)
    
    denom=sum([np.exp((-(number-n)**2)/ (2*sigma**2)) for number in range(n+1)]) 
    print(denom)
    
    print(num/denom)
    return num/denom
    

def smooth(x): #TODO: figure out good params for the smoothing
    #So there are 15 obs/sec and I think the hoover paper smoothed minute by minute
    #not really about sigma or window size, or end behavior     
    y=x    
    

    
    for col in [0, 1, 2, 3, 5, 6]: # skip the time index
        y[:,col]=scipy.ndimage.filters.gaussian_filter1d(x[:,col], 10)#maybe replace this with the other helper function
        
        
        #this is from the paper
        #might need to have this be a per thing basis

    smoothed_df=pd.DataFrame(y)
    smoothed_df.to_csv(subj+"_smoothed.csv")    
    
    print("I smoothed some data")
    return y
    
def energyGeneration(x): #maybe change this to include Time?
    #energy goes first (look at eq 2 from paper)    
    window_size=720
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
    t=0
    start_segment=0
    #Use a subset for now!
#    energy=energy.loc[:]
  
    
    plt.plot(energy["Energy"].iloc[:])    
    plt.show()    
    
    peak_dictionary={"time":[],"value":[]}
        
    print(energy.shape[0])
    while t < energy.shape[0]: 
#        print("the t value is currently:", t)
#        print("the energy value is currently:", energy["Energy"].iloc[t])
        thresh1=energy["Energy"].iloc[t]
        thresh2=thresh1*2 #originally this is 2
        
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
       
        duration_t=t #FIXME: this is probs the bug
        
        while(energy["Energy"].iloc[duration_t] >= thresh1 and duration_t < energy.shape[0] -1 ):
            duration_t+=1
        print("I think the peak is between", local_t)
        print("and it's duration is until time", duration_t)
        t=duration_t
        
        local_peak=max(energy["Energy"].iloc[start_segment:duration_t])
             
        temp_df=pd.DataFrame(columns=["time_of_peak","peak_value"])
        hmm=energy["Energy"].iloc[start_segment:duration_t].argmax()
        print(hmm) 
        print("this should be the index",hmm)

        #times part is not right maybe have a +t because argmax might be from the start of the interval
        peak_dictionary["time"].append(hmm)
        peak_dictionary["value"].append(local_peak)
        
        #maybe update the start of the buffer here?
        start_segment=t        
        
        t+=1
    
    peaks=pd.DataFrame(peak_dictionary)
#    peaks.to_csv(path+subj+"_peaks.csv")
    
    return peaks
    
def featureGeneration(smooth, peaks):
    #do some stuff to shape according to the peaks and then have a for loop
    #in each for loop, they should add there stuff to a csv
    
    #TODO: this first part
    #smooth + peaks -> data
    number_segments=peaks.shape[0]+1    
    
    #probably should add what the segment is
    features=pd.DataFrame(np.zeros((number_segments,4)), columns=["LinearAcc","Manipulation", "WristRoll","WristRollRegularity"])
    
    start_t=0    
    iter=0
    for t in peaks["time"]:
        segment=smooth.loc[start_t:t]
        features["Manipulation"].iloc[iter]=manipulationFeature(segment)
        features["LinearAcc"].iloc[iter]=linearAccelerationFeature(segment)
        features["WristRoll"].iloc[iter]=wristRollFeature(segment)
        features["WristRollRegularity"].iloc[iter]=wristRollRegularityFeature(segment)
        print("finshed with segment",iter )
        iter+=1
        start_t=t

    features.to_csv(path+subj+"_features.csv")
    
    return features
    
def manipulationFeature(segment):
    num=[abs(number) for number in segment["Angular_Velocity_x"]]+[abs(number) for number in segment["Angular_Velocity_y"]]+[abs(number) for number in segment["Angular_Velocity_z"]]
    denom=[abs(number) for number in segment["Linear_Accel_x"]]+[abs(number) for number in segment["Linear_Accel_y"]]+[abs(number) for number in segment["Linear_Accel_z"]]
    
    manip=1/(len(segment))*sum(np.divide(num,denom)) # I really hope this is elementwise
    #also hope this is in degrees per second
    
    return manip
    
def linearAccelerationFeature(segment):
    linacc=sum([abs(number) for number in segment["Linear_Accel_y"]])+sum([abs(number) for number in segment["Linear_Accel_z"]])
    linacc=1/(len(segment)) *  linacc #this might have to be shape[0] insteas of len
    return linacc
  
def wristRollFeature(segment):
    #double check to make sure this is supposed to be y
    wr=1/(len(segment)) * sum([abs(number - np.average(segment["Angular_Velocity_y"])) for number in segment["Angular_Velocity_y"]])
    
    return wr
    
def wristRollRegularityFeature(segment):
    #I don't get the whole +8 thing
    #TODO: fix this with the + 8 seconds
    return 1/(len(segment))*sum([1 for number in segment["Angular_Velocity_y"] if abs(number) > 10 ])


def classification(feats,target):
    gnb=GaussianNB()
    
    gnb.fit(features,target)
    
    print(gnb.predict_proba(features))
    
    #raise Exception("Not impletemented")

def makePlot(peaks,energy):
     plt.plot(peaks["time"],peaks["value"], marker='o', linestyle='None', color='r')
     plt.plot(energy["Energy"])
     plt.show()
     plt.savefig("annotatedEnergy.png" )

if __name__ == "__main__":
    if not os.path.exists(path+subj+"_smoothed.csv"):
        raw=readData(path+file_name)
        smoothed=smooth(raw)
    else:
        smoothed=pd.read_csv(subj+"_smoothed.csv",index_col=0, header=0,names=["Linear_Accel_x","Angular_Velocity_x","Linear_Accel_y","Time","Angular_Velocity_z","Angular_Velocity_y","Linear_Accel_z"])
    if not os.path.exists(path+subj+"_energy.csv"):
        energy=energyGeneration(smoothed) #TODO: something with the os to only do this if it doesn't exist
    else:
        energy=pd.read_csv(subj+"_energy.csv", names=["Energy"],index_col=0, header=0)
    if not os.path.exists(path+subj+"_peaks.csv"):
        # use my own function to see what is happening
        energy=pd.DataFrame(np.multiply(np.arange(100),2*np.sin(0.1*np.arange(100))**2),columns=["Energy"])  
        peaks=hooverSegmentation(energy)
    else:
        peaks=pd.read_csv(path+subj+"_peaks.csv", names=["time",'value'],index_col=0, header=0)
      
    if not os.path.exists(path+subj+"_peaks.csv"):
        features=featureGeneration(smoothed,peaks)
    else:
        features=pd.read_csv(path+subj+"_features.csv",index_col=0, header=0,names=["LinearAcc","Manipulation", "WristRoll","WristRollRegularity"])
    
    #smoothed["Time"].iloc[peaks["time"]] this is not in the right spot, but it might be good
    
    #these targets are not right but just go with it
    targets=pd.read_csv(path+subj+"_targets.csv",header=0,names=["labels"]) #change this to a function that reads in the episode time and calculates if its in the segement
    
    
#    classification(features,targets)
    
    makePlot(peaks,energy)#also add the actual eating epsiodes 
    
    
    print("yay done with main!")
    