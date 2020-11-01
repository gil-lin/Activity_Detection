# -*- coding: utf-8 -*-

import glob
import os
import numpy as np
import pandas as pd

def clean_and_save(file):
        df = pd.read_csv(file).tail(-3)
        columns = df.iloc[0,0].split()
        columns = columns[:-5]
        
        temp  = [[] for x in range(len(columns))]
        dfObj = pd.DataFrame(columns=columns)
        df = df.tail(-1)
        rows = df.shape[0]

        for row in range(1,rows-1):
            s = df.iloc[row,0]
            s = np.array(s.split())
            if len(s)==4:
                continue
            factor = 23 - len(s)
            s = s[:(-5+factor)]
            
            for i in range(len(columns)):
                temp[i].append(s[i])
        for i in range(len(columns)):
            dfObj[columns[i]] = temp[i]
        dfObj.drop(columns[:9],axis = 1, inplace=True) 
        return dfObj
        
# take only the chest dataset
directory = 'C:\\Users\\eyaraz\\Desktop\\Deep_Learning_A_Z\\fall_detector\\Kaggle\\simulated_falls-and-DLAs\\'
directory_Falls = 'C:\\Users\\eyaraz\\Desktop\\Deep_Learning_A_Z\\fall_detector\\Kaggle\\Chesat_Falls\\'
directory_ADL = 'C:\\Users\\eyaraz\\Desktop\\Deep_Learning_A_Z\\fall_detector\\Kaggle\\Chesat_ADL\\'
dirs = [x[0] for x in os.walk(directory)]
counter_fall = counter_ADL = 0
for path in dirs:
    # print(path)
    path = path+"\\"
    for file in glob.glob(path+"\\" + "*.txt"):
        number_sensor =int(np.array([file.split('\\')[-1][:6]]))
        if number_sensor == 340527:
            # df = pd.read_csv(file)
            df = clean_and_save(file)
            number_fall = int(np.array([file.split('\\')[-3][0]])) #8 or 9 
            if number_fall == 9:
                df.to_csv(directory_Falls + "Fall"+str(counter_fall)+".csv")
                counter_fall +=1
            else:
                df.to_csv(directory_ADL + "ADL"+str(counter_ADL)+".csv")
                counter_ADL +=1
                
                
                
                
# convert all samples to the same len (500)              
path_to_read = 'Kaggle\\Chest_dataset\\Chest_Falls\\dataset\\'
path_to_write = 'Kaggle\\clean_Chest_dataset\\Chest_Fall\\dataset\\'
counter_ADL = 0
for file in glob.glob(path_to_read + "*.csv"):
    filename = file.split('\\')[2]
    df = pd.read_csv(file)       
    df.drop('Unnamed: 0', axis=1, inplace = True)
    rows = df.shape[0]

    if rows>=500:
        df = df.loc[int(rows/2)-250 : int(rows/2)+249]
    else:
        num_first = int((500-rows) / 2)
        num_end = 500- rows - num_first
        temp_end = np.stack([np.array(df)[-1]]*num_end)
        df_temp_end = pd.DataFrame(data = temp_end, columns=df.columns)
        df = pd.concat([df,df_temp_end])
        if num_first > 0:
            temp_first = np.stack([np.array(df)[0]]*num_first)
            df_temp_first = pd.DataFrame(data = temp_first, columns=df.columns)
            df = pd.concat([df_temp_first,df])
            
    df = df.reset_index(drop = True)        
    df.to_csv(path_to_write + "Fall"+str(counter_ADL)+".csv")
    counter_ADL +=1  