# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy

def save_plots(df_ac,df_gy,df_mg,filename, scenario):

    acllemoter_X ,acllemoter_Y, acllemoter_Z = df_ac["X-Axis"], df_ac["Y-Axis"], df_ac["Z-Axis"]
    samples_ac = df_ac["Sample_No"]

    gyro_X, gyro_Y, gyro_Z = df_gy["X-Axis"], df_gy["Y-Axis"], df_gy["Z-Axis"]
    samples_gy = df_gy["Sample_No"]
    
    magnet_X, magnet_Y, magnet_Z = df_mg["X-Axis"], df_mg["Y-Axis"], df_mg["Z-Axis"]
    samples_mg = df_mg["Sample_No"]
    fig = plt.figure()

    plt.subplot(3, 3, 1)
    plt.plot(samples_ac, acllemoter_X, 'b-', label='X-axis')
    plt.ylabel('X_' + scenario)
    
    plt.subplot(3, 3, 2)
    plt.plot(samples_ac, acllemoter_Y, 'r-', label='Y-axis')
    plt.ylabel('Y_'+ scenario)
    
    plt.subplot(3, 3, 3)
    plt.plot(samples_ac, acllemoter_Z, 'g-', label='Z-axis')
    plt.ylabel('Z_'+ scenario)
    
    plt.subplot(3, 3, 4)
    plt.plot(samples_gy, gyro_X, 'b-', label='X-axis')
    plt.ylabel('X_' + scenario)
    
    plt.subplot(3, 3, 5)
    plt.plot(samples_gy, gyro_Y, 'r-', label='Y-axis')
    plt.ylabel('Y_'+ scenario)
    
    plt.subplot(3, 3, 6)
    plt.plot(samples_gy, gyro_Z, 'g-', label='Z-axis')
    plt.ylabel('Z_'+ scenario)

    plt.subplot(3, 3, 7)
    plt.plot(samples_mg, magnet_X, 'b-', label='X-axis')
    plt.ylabel('X_' + scenario)
    
    plt.subplot(3, 3, 8)
    plt.plot(samples_mg, magnet_Y, 'r-', label='Y-axis')
    plt.ylabel('Y_'+ scenario)
    
    plt.subplot(3, 3, 9)
    plt.plot(samples_mg, magnet_Z, 'g-', label='Z-axis')
    plt.ylabel('Z_'+ scenario)
    
    fig.tight_layout()
    
    fig.savefig('graphs\\'+scenario+'\\'+filename+'.png')
    plt.close(fig)

def save_plots_Kaggle(mesure_array, filename, path_to_write, columns, fft_mode=False):
    samples = mesure_array[0]
    fig = plt.figure()
    N = len(samples)
    for i in range(1,len(mesure_array)):
        if fft_mode:
            mesure_array[i] -= mesure_array[i].mean()
            mesure_array[i] = 1/N * np.abs(scipy.fft(mesure_array[i]))
        plt.subplot(3, 3, i)
        plt.plot(samples, mesure_array[i], 'b-', label=columns[i][-1]+'-axis')
        if fft_mode:
            plt.ylabel("fft"+columns[i]) 
        else:
            plt.ylabel(columns[i]) 
    fig.tight_layout()
    fig.savefig(path_to_write+filename+'.png')
    plt.close(fig)
       
def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_features(X_adl, X_fall, labels, axis_name, width=0.35):
    x = np.arange(len(labels))
    count = 0
    # X_adl /= X_adl.std(axis=0)
    # X_fall /= X_fall.std(axis=0)   
    
    #mean
    X_adl_mean = X_adl.mean(axis=0)
    X_fall_mean = X_fall.mean(axis=0)
    #std
    X_adl_std = X_adl.std(axis=0)
    X_fall_std = X_fall.std(axis=0)    
    

    for label in labels:
        fig, ax = plt.subplots()

        rects1 = ax.bar(0-width/2, X_adl_mean[count], width, yerr=X_adl_std[count], label='adl')
        rects2 = ax.bar(0+width/2, X_fall_mean[count], width, yerr=X_fall_std[count],
                        label='fall')
        
        ax.set_ylabel('Scores')
        ax.set_title('{}:Scores by {}'.format(axis_name,label))
        # ax.set_xticks(x)
        # ax.set_xticklabels(label)
        ax.legend()
        
        autolabel(rects1,ax)
        autolabel(rects2,ax)
        
        fig.tight_layout()
        
        plt.show()
        count+=1
    
# file_ADL = 'Dataset/cleast_accelometer/UMAFall_Subject_01_ADL_Aplausing_1_2017-04-14_23-38-23.csv'
# file_Fall = 'Fall.csv'
# file_Fall = 'Dataset/cleast_accelometer/UMAFall_Subject_02_Fall_forwardFall_5_2016-06-13_20-54-39.csv'

mode =2

path_to_read = 'Dataset\\clean_Dataset\\'

# df_ADL = pd.read_csv(file_ADL)
# df_Fall = pd.read_csv(file_Fall)


# acllemoter_X_ADL = df_ADL["X-Axis"]
# acllemoter_Y_ADL = df_ADL["Y-Axis"]
# acllemoter_Z_ADL = df_ADL["Z-Axis"]
# samples = df_ADL["Sample_No"]

# acllemoter_X_Fall = df_Fall["Acc_X"]
# acllemoter_Y_Fall = df_Fall["Acc_Y"]
# acllemoter_Z_Fall = df_Fall["Acc_Z"]
# samples = df_Fall["Unnamed: 0"]

#_____________________________________________________________________________

#_____________________________________________________________________________

if mode ==1:
        
    for file in glob.glob(path_to_read + "*.csv"):
        filename = file.split('\\')[2]
        
        df = pd.read_csv(file)
        filt_ID = (df["Sensor ID"] ==1)
        #ac
        filt_type_ac = (df["Sensor Type"] ==0)
        df_ac = df.loc[filt_ID].loc[filt_type_ac]
        #gy
        filt_type_gy = (df["Sensor Type"] ==1)
        df_gy = df.loc[filt_ID].loc[filt_type_gy]
        #mg
        filt_type_mg = (df["Sensor Type"] ==2)
        df_mg = df.loc[filt_ID].loc[filt_type_mg]
        if filename[19] == 'A':
            scenario = "ADL"
        else:
            scenario = "FALL"
        save_plots(df_ac, df_gy, df_mg, filename, scenario)
        
        
elif mode ==2:
        path_to_read = 'Kaggle\\clean_Chest_dataset\\Chest_Fall\\dataset\\'
        path_to_write = 'Kaggle\\clean_Chest_dataset\\Chest_Fall\\graphs\\'
        fft_mode = True
        for file in glob.glob(path_to_read + "*.csv"):
            filename = file.split('\\')[-1].split('.')[0]
            df = pd.read_csv(file)            
            mesurement_array = [[] for x in range(10)]
            j = 0
            for col in df.columns:
                mesurement_array[j] = np.array(df[col])
                j+=1
            save_plots_Kaggle(mesurement_array, filename, path_to_write, df.columns, fft_mode = False)
            
            
            
            
elif mode ==3:
    n_data, n_features, axis = X.shape
    X_adl, X_fall = X[:241],X[241:]
    labels = index
    for ax in range(axis):
        axis_name = df.columns[ax]
        plot_features(X_adl[:,:,ax], X_fall[:,:,ax], labels, axis_name, width=0.35)      