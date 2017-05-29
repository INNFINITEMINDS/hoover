# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:27:34 2017

implementing Adam Hoover's method 
@author: Matt
"""
import csv
import numpy as np
import pandas as pd
import datetime
from scipy import signal
import scipy.ndimage
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB

import os

path="P14/" #../../MotifCounter/
subj="P14" #tried P5 and I don't think there was enough variability to do this
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
    
def smoothing(dataDf, method="sliding", **kwargs):

    """smoothing function including sliding smoothing(box smoothing) and gaussian filter
    Parameters
    ----------
        dataDf:                 dataFrame
        method:                 str.  "sliding" or "gaussian"
        kwargs:
          when method == "sliding":
            winsize:            int
          when method == "gaussian":
            winsize:            int
            sigma:              int
    Return
    ------
        dataDf:                 dataFrame
    """
    if method == "sliding":
        if 'winsize' in kwargs:
            winsize = kwargs['winsize']
        else:
            winsize = 10
            print("Arg 'winsize' is set as default ", winsize)
            
        dataDf = dataDf.rolling(window=winsize, center=False).mean()
        dataDf = dataDf.fillna(method = 'backfill')

    # TODO: implement gaussian smoothing method according to the requirement
    elif method == "gaussian":

        if 'winsize' in kwargs:
            winsize = kwargs['winsize']
            print('window size: ',  winsize)
        else:
            winsize = 10
            print("Arg 'winsize' is set as default ", winsize)

        if 'sigma' in kwargs:
            sigma = kwargs['sigma']
            print('sigma: ', sigma)
        else:
            sigma = 7
            print("Arg 'sigma' is set as default ", sigma)

#        names = list(dataDf.columns.values)
        arr = dataDf.as_matrix()
        arr=np.reshape(arr,(arr.shape[0],1))
        for c in range(arr.shape[1]):
            col = arr[:,c]
            pad = [0] * (winsize-1)
        
            s=np.r_[pad,col]
            w = eval('signal.'+method+'('+str(winsize)+','+str(sigma)+ ')')
            w[32:]=0
#            print(w)
            
            smoothed=np.convolve(w/w.sum(),s,mode='valid')
            arr[:,c] = smoothed

#        dataDf = pd.DataFrame(data = arr, columns = names)
    
    arr=np.reshape(arr, (arr.shape[0],))
    return arr

def smooth(x): #TODO: figure out good params for the smoothing
    #So there are 15 obs/sec and I think the hoover paper smoothed minute by minute
    #not really about sigma or window size, or end behavior     
    y=x    
    
    for col in ["Linear_Accel_x", "Angular_Velocity_x", "Linear_Accel_y", "Angular_Velocity_z", "Angular_Velocity_y","Linear_Accel_z"]: #FIXME: skip the time index
#       y[col].iloc[:]=scipy.ndimage.filters.gaussian_filter1d(x[col].iloc[:], 10)#maybe replace this with the other helper function
        print("now smoothing ",col)
#        y[col].iloc[:]=gauss(x[col].iloc[:],3,10)
        y[col].iloc[:]=smoothing(x[col].iloc[:], method="gaussian", sigma=10, winsize=62)
        
        #this is from the paper
        #might need to have this be a per thing basis

    smoothed_df=pd.DataFrame(y)
    smoothed_df.to_csv(path+subj+"_smoothed.csv")    
    
    print("I smoothed some data")
    return smoothed_df
    
def energyGeneration(x): 
    #energy goes first (look at eq 2 from paper)    
    window_size=360#1860#360#720
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

    plt.plot(energy["Energy"].iloc[:])    
    plt.show()    
    
    peak_dictionary={"time":[],"value":[]}
        
    print(energy.shape[0])
    while t < energy.shape[0]: 
        #there is a weird edge case where this is 0, so both thresh are 0
        thresh1=energy["Energy"].iloc[t]
        thresh2=thresh1*1.01 #originally this is 2
#        print("the thresholds are",thresh1,thresh2)
        
        local_t=t
        while(energy["Energy"].iloc[local_t] < thresh2 and local_t < energy.shape[0] -1):
            local_t+=1
            if energy["Energy"].iloc[local_t] < thresh1:
                #reset the thresholds
                print("reseting the threshold at index",local_t)
                thresh1=energy["Energy"].iloc[local_t]
                thresh2=thresh1*1.1
                
        t=local_t
        duration_t=t 
        
        while(energy["Energy"].iloc[duration_t] > thresh1 and duration_t < energy.shape[0] -1 ):
            duration_t+=1

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
    smooth=smooth.set_index(["Time"])
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

def getGroundTruth(filename): #FIXME: might have to add the whole hour offset thing!/deal with time zone
    df=pd.read_csv(filename)
    out_df=df
    for ii in range(df["StartTime"].shape[0]):
        for s in ["StartTime","EndTime"]:
            try:
                dtDate=datetime.datetime.strptime(df[s].iloc[ii], "%Y-%m-%d %H:%M:%S.%f")
            except:
                dtDate=datetime.datetime.strptime(df[s].iloc[ii], "%Y-%m-%d %H:%M:%S")
            utime=dtDate.timestamp()*1000
            print(utime)
            
            out_df[s].iloc[ii]=utime
    
    return out_df    
    
def makeSignalPlot(signal,title):
    plt.title(title)
    plt.plot(signal["Time"],signal["Linear_Accel_x"])
    plt.plot(signal["Time"],signal["Linear_Accel_y"])
    plt.plot(signal["Time"],signal["Linear_Accel_z"])
    plt.show()
    
def makePlot(peaks,energy,gtruth):
    plt.title("Energy vs. Unix time, with highlighted eating episodes")
    plt.plot(peaks["time"],peaks["value"], marker='o', linestyle='None', color='r')
    plt.plot(energy["Time"],energy["Energy"])
    for i in range(gtruth.shape[0]):
        plt.axvspan(xmin=gtruth["StartTime"].iloc[i],xmax=gtruth["EndTime"].iloc[i], color='red', alpha=0.3)
        
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(path+subj+"_smoothed.csv"):
        raw=readData(path+file_name)
        makeSignalPlot(raw,"raw")
        smoothed=smooth(raw)
    else:
        smoothed=pd.read_csv(path+subj+"_smoothed.csv",index_col=0, header=0,names=["Linear_Accel_x","Angular_Velocity_x","Linear_Accel_y","Time","Angular_Velocity_z","Angular_Velocity_y","Linear_Accel_z"])
    makeSignalPlot(smoothed,"smoothed")    
    
    if not os.path.exists(path+subj+"_energy.csv"):
        energy=energyGeneration(smoothed)
    else:
        energy=pd.read_csv(path+subj+"_energy.csv", names=["Energy", "Time"],index_col=0, header=0)
    
    if not os.path.exists(path+subj+"_peaks.csv"):
        # use my own function to see what is happening
#        energy=pd.DataFrame(np.multiply(np.arange(100),2*np.sin(.1*np.arange(100))**2)+10-0.1*np.arange(100),columns=["Energy"])  
        #
        peaks=hooverSegmentation(energy)
    else:
        peaks=pd.read_csv(path+subj+"_peaks.csv", names=["time",'value'],index_col=0, header=0)

#        peaks=hooverSegmentation(energy.loc[energy["Time"]>1477151816884.0].loc[ energy["Time"]<1477165667259.0]) #hard code for now
      
    if not os.path.exists(path+subj+"_features.csv"):
        features=featureGeneration(smoothed,peaks)
    else:
        features=pd.read_csv(path+subj+"_features.csv",index_col=0, header=0,names=["LinearAcc","Manipulation", "WristRoll","WristRollRegularity"])
    
    #smoothed["Time"].iloc[peaks["time"]] this is not in the right spot, but it might be good
    
    #these targets are not right but just go with it
#    targets=pd.read_csv(path+subj+"_targets.csv",header=0,names=["labels"]) #change this to a function that reads in the episode time and calculates if its in the segement
    
    
#    classification(features,targets)
    
    
    truth=getGroundTruth(path+subj+"_gestures.csv") #TODO: do the 
    makePlot(peaks,energy, truth)#also add the actual eating epsiodes 
    
    
    print("yay done with main!")
    