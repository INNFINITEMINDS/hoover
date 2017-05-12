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

path="P4/" #../../MotifCounter/
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
    
    X_df=pd.DataFrame(X,columns=["Linear_Accel_x","Angular_Velocity_x","Linear_Accel_y","Time","Angular_Velocity_z","Angular_Velocity_y","Linear_Accel_z"	])
    print("done reading in the data")
    return X_df

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
    
    for col in ["Linear_Accel_x", "Angular_Velocity_x", "Linear_Accel_y", "Angular_Velocity_z", "Angular_Velocity_y","Linear_Accel_z"]: #FIXME: skip the time index
       y[col].iloc[:]=scipy.ndimage.filters.gaussian_filter1d(x[col].iloc[:], 10)#maybe replace this with the other helper function
        
        
        #this is from the paper
        #might need to have this be a per thing basis

    smoothed_df=pd.DataFrame(y)
    smoothed_df.to_csv(path+subj+"_smoothed.csv")    
    
    print("I smoothed some data")
    return smoothed_df
    
def energyGeneration(x): 
    #energy goes first (look at eq 2 from paper)    
    window_size=720#1860#360#720
    #Naively try 20 seconds 720
    #1860 #they use 1 minute, so 31hz*60 that many obs per min
    
    #cut around the part where they shake their arms
    energy_df=pd.DataFrame(np.zeros((x.shape[0]-window_size,2)) , columns=["Energy","Time"])
    
    print("this thing should happen times",x.shape[0]-window_size)    
    
    for ii in range(x.shape[0]-window_size): 
        sumx=sum([abs(number) for number in x["Linear_Accel_x"].iloc[int(window_size/2)+ii:window_size+ii]])
        sumy=sum([abs(number) for number in x["Linear_Accel_y"].iloc[int(window_size/2)+ii:window_size+ii]])
        sumz=sum([abs(number) for number in x["Linear_Accel_z"].iloc[int(window_size/2)+ii:window_size+ii]])
        energy_df["Energy"].iloc[ii]=1/(window_size+1)*(sumx+sumy+sumz)
        
        energy_df["Time"].iloc[ii]=x["Time"].iloc[ii]
        if(ii%1000==1):
            print("I am on number",ii)
            assert energy_df["Time"].iloc[ii] != energy_df["Time"].iloc[ii-1]
            
    energy_df.to_csv(path+subj+"_energy.csv")
    
    
    return energy_df
    
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
        #there is a weird edge case where this is 0, so both thresh are 0
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
        
        while(energy["Energy"].iloc[duration_t] > thresh1 and duration_t < energy.shape[0] -1 ):
            duration_t+=1
        #maybe this shouldnt be here?
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
        peak_dictionary["time"].append(energy["Time"].iloc[hmm])
        peak_dictionary["value"].append(local_peak)
        
        #maybe update the start of the buffer here?
        start_segment=t        
            
    peaks_df=pd.DataFrame(peak_dictionary)
    peaks_df.to_csv(path+subj+"_peaks.csv")
    
    return peaks_df
    
def featureGeneration(smooth, peaks):
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
    
    manip=1/(len(segment))*sum(np.divide(num,denom))
    
    return manip
    
def linearAccelerationFeature(segment):
    linacc=sum([abs(number) for number in segment["Linear_Accel_y"]])+sum([abs(number) for number in segment["Linear_Accel_z"]])
    linacc=1/(len(segment)) *  linacc #this might have to be shape[0] insteas of len
    return linacc
  
def wristRollFeature(segment):
    wr=1/(len(segment)) * sum([abs(number - np.average(segment["Angular_Velocity_y"])) for number in segment["Angular_Velocity_y"]])
    
    return wr
    
def wristRollRegularityFeature(segment):
    #TODO: fix this with the + 8 seconds
    return 1/(len(segment))*sum([1 for number in segment["Angular_Velocity_y"] if abs(number) > 10 ])


def classification(feats,target):
    gnb=GaussianNB()
    gnb.fit(feats,target)
    print(gnb.predict_proba(feats))

def getGroundTruth(filename):
    #TODO: actually implement this
    #this is a list of a list of start and endtimes
    eating_episodes=[[1477152091713.0, 1477152414276.0],[1477155000184.0,1477156616494.0],[1477163082255.0,1477162228768.0]]
    
    return eating_episodes
    
    
def makeSignalPlot(signal,title):
    plt.title(title)
    plt.plot(signal["Time"],signal["Linear_Accel_x"])
    plt.plot(signal["Time"],signal["Linear_Accel_y"])
    plt.plot(signal["Time"],signal["Linear_Accel_z"])
    plt.show()
    
def makePlot(peaks,energy,gtruth):
    plt.plot(peaks["time"],peaks["value"], marker='o', linestyle='None', color='r')
    plt.plot(energy["Time"],energy["Energy"])
    #FIXME: change the line below to deal with the annotations
    #plt.axvspan(3, 6, color='red', alpha=0.5)
    for i in range(len(gtruth)):
        plt.axvspan(xmin=gtruth[i][0],xmax=gtruth[i][1], color='red', alpha=0.3)
        
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(path+subj+"_smoothed.csv"):
        raw=readData(path+file_name)
        makeSignalPlot(raw,"raw")
        smoothed=smooth(raw)
    else:
        smoothed=pd.read_csv(path+subj+"_smoothed.csv",index_col=0, header=0,names=["Linear_Accel_x","Angular_Velocity_x","Linear_Accel_y","Time","Angular_Velocity_z","Angular_Velocity_y","Linear_Accel_z"])
        smoothed= smoothed.iloc[1000:30000]#I'm fucking with this atm 
    makeSignalPlot(smoothed,"smoothed")    
    
    if not os.path.exists(path+subj+"_energy.csv"):
        energy=energyGeneration(smoothed)
    else:
        energy=pd.read_csv(path+subj+"_energy.csv", names=["Energy", "Time"],index_col=0, header=0)
    
    if not os.path.exists(path+subj+"_peaks.csv"):
        # use my own function to see what is happening
#        energy=pd.DataFrame(np.multiply(np.arange(100),2*np.sin(.1*np.arange(100))**2)+10-0.1*np.arange(100),columns=["Energy"])  
        peaks=hooverSegmentation(energy)
    else:
        peaks=pd.read_csv(path+subj+"_peaks.csv", names=["time",'value'],index_col=0, header=0)
      
#    if not os.path.exists(path+subj+"_features.csv"):
#        features=featureGeneration(smoothed,peaks)
#    else:
#        features=pd.read_csv(path+subj+"_features.csv",index_col=0, header=0,names=["LinearAcc","Manipulation", "WristRoll","WristRollRegularity"])
    
    #smoothed["Time"].iloc[peaks["time"]] this is not in the right spot, but it might be good
    
    #these targets are not right but just go with it
#    targets=pd.read_csv(path+subj+"_targets.csv",header=0,names=["labels"]) #change this to a function that reads in the episode time and calculates if its in the segement
    
    
#    classification(features,targets)
    
    
    truth=getGroundTruth(path+subj)
    makePlot(peaks,energy, truth)#also add the actual eating epsiodes 
    
    
    print("yay done with main!")
    