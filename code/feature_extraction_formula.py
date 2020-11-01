# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import kurtosis, skew
import time
from scipy.stats import entropy

class feature_extraction:
    
    def __init__(self, X):
        self.X = X
        
    def feature_extraction(self,
                           flag_mean = False, flag_std = False,
                           flag_med = False,flag_max = False,
                           flag_min = True,flag_iqr = False,
                           flag_sk = False,flag_kurt = False,
                           flag_amv = False,flag_var = False,
                           flag_rms = False,flag_energy = True,
                           flag_sra = False,flag_pp = False,
                           flag_cf = True,flag_if = False,
                           flag_mf = False,flag_sf = False,
                           flag_cross_correlation = False):
                           
        tic = time.time()
        index = []
        X_features = []
        cross_correlation = None
        num_data, len_signal, channels  = self.X.shape
        
        # mean
        if flag_mean:
            X_mean = np.reshape(self.X.mean(axis = 1),(num_data,1,channels))
            X_features.append(X_mean)
            index.append('mean')
        # standard deviation
        if flag_std:
            X_std = np.reshape(self.X.std(axis = 1), (num_data,1,channels))
            X_features.append(X_std)
            index.append('std')
        # median
        if flag_med:
            X_med = np.reshape(np.median(self.X,axis = 1), (num_data,1,channels))
            X_features.append(X_med)
            index.append('median')
        # maximum
        if flag_max:
            X_max = np.reshape(self.X.max(axis = 1), (num_data,1,channels))
            X_features.append(X_max)
            index.append('max')
        # minimum
        if flag_min:
            X_min = np.reshape(self.X.min(axis = 1), (num_data,1,channels))
            X_features.append(X_min)
            index.append('min')
        # iqr - interquartile range
        if flag_iqr:
            q75, q25 = np.percentile(self.X, [75 ,25],axis = 1)
            X_iqr = np.reshape(q75-q25, (num_data,1,channels))
            X_features.append(X_iqr)
            index.append('iqr')
        # correlation coefficient - same to "cross correlation"
        #-----------------------------------------------------        
        # skewness
        if flag_sk:
            X_sk = np.reshape(skew(self.X, axis = 1), (num_data,1,channels))
            X_features.append(X_sk)
            index.append('skewness')
        # Kurtosis
        if flag_kurt:
            X_ku = np.reshape(kurtosis(self.X, axis = 1), (num_data,1,channels))
            X_features.append(X_ku)
            index.append('kurtosis')
        # cross correlation
        if flag_cross_correlation:
            cross_correlation = []
            X_mean = np.reshape(self.X.mean(axis = 1),(num_data,1,channels))
            X_normalize = self.X - X_mean
            X_normalize /= np.reshape(self.X.std(axis = 1), (num_data,1,channels))
            for ax1 in range(channels):
                for ax2 in range(ax1+1,channels):
                    cross_correlation.append((X_normalize[:,:,ax1] * X_normalize[:,:,ax2]).sum(axis=1))        
            cross_correlation = np.array(cross_correlation).T
        # absulute mean value
        if flag_amv:
            X_amv = np.reshape((np.abs(self.X)).mean(axis=1), (num_data,1,channels))
            X_features.append(X_amv)
            index.append('absolute mean value')
        # variance
        if flag_var:
            X_std = np.reshape(self.X.std(axis = 1), (num_data,1,channels))
            X_var = X_std ** 2
            X_features.append(X_var)
            index.append('var')
        # rms - root mean squre
        if flag_rms:    
            X_rms = np.reshape(np.linalg.norm(self.X, axis=1)/len_signal, (num_data,1,channels))
            X_features.append(X_rms*10)
            index.append('rms')
        # Energy
        if flag_energy: 
            X_energy = np.reshape((np.linalg.norm(self.X, axis=1) ** 2)/len_signal , (num_data,1,channels))
            X_features.append(X_energy)
            index.append('energy')
        # SRA
        if flag_sra:
            X_sra = np.reshape(np.sqrt(np.abs(self.X)).mean(axis=1), (num_data,1,channels))
            X_features.append(X_sra*10)
            index.append('SRA')
        # p_p - peak to peak
        if flag_pp:      
            X_max = np.reshape(self.X.max(axis = 1), (num_data,1,channels))
            X_min = np.reshape(self.X.min(axis = 1), (num_data,1,channels))
            X_pp = X_max - X_min
            X_features.append(X_pp)
            index.append('p_p')
        # crest factor
        if flag_cf: # X_max / X_l2
            X_l2 = np.reshape(np.linalg.norm(self.X, axis=1), (num_data,1,channels))
            X_cf = np.reshape(np.abs(self.X).max(axis=1), (num_data,1,channels)) / X_l2
            X_features.append(X_cf*10)
            index.append('crest factor')
        # impulse factor
        if flag_if:
            X_if = np.reshape(np.abs(self.X).max(axis=1), (num_data,1,channels)) / X_mean
            X_features.append(X_if)
            index.append('impulse factor')
        # margin factor
        if flag_mf:
            X_sra = np.reshape(np.sqrt(np.abs(self.X)).mean(axis=1), (num_data,1,channels))
            X_mf = np.reshape(np.abs(self.X).max(axis=1), (num_data,1,channels)) / X_sra
            X_features.append(X_mf)
            index.append('margin factor')
        # shape factor
        if flag_sf:
            X_rms = np.reshape(np.linalg.norm(self.X, axis=1)/len_signal, (num_data,1,channels))
            X_amv = np.reshape((np.abs(self.X)).mean(axis=1), (num_data,1,channels))
            X_sf = X_rms / X_amv
            X_features.append(X_sf)
            index.append('shape factor')
        # frequency center
        
        # rms frequency
        
        # rvf - root variant frequency
        
        #entropy
    
        # Concatenate
    
        X_featurs = np.concatenate([feat for feat in X_features], axis=1)
        toc = time.time()
        # print("feature extraction function takes {} sec".format(toc-tic))
        return  X_featurs, index, cross_correlation