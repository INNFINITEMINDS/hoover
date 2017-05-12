# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:54:24 2017

@author: Matt
"""

import numpy as np
import pandas as pd

path=
subj=


def featureGeneration(smooth, peaks):
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
    linacc=1/(len(segment)) *  linacc
    return linacc
  
def wristRollFeature(segment):
    wr=1/(len(segment)) * sum([abs(number - np.average(segment["Angular_Velocity_y"])) for number in segment["Angular_Velocity_y"]])
    
    return wr
    
def wristRollRegularityFeature(segment):
    return 1/(len(segment))*sum([1 for number in segment["Angular_Velocity_y"] if abs(number) > 10 ])