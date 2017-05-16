# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:23:55 2017

@author: Matt
"""

import matplotlib.pyplot as plt 
import numpy as np
import datetime
import pandas as pd

def toyPlot(eating_times):
    plt.plot(np.arange(10), marker='o', linestyle='None', color='b')
    for i in range(len(eating_times)):
        plt.axvspan(xmin=eating_times[i][0],xmax=eating_times[i][1], color='red', alpha=0.3)
    plt.show()

def convertTimeToBandTime():
    df=pd.read_csv("P4/P4_gestures.csv")
    out_df=df
#    dtDate = datetime.datetime.strptime("07/27/2012","%m/%d/%Y")
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
    
if __name__=="__main__":
    toyPlot([[0,.7],[3,5],[6.4,8]])
    y=convertTimeToBandTime()