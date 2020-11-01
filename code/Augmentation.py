# -*- coding: utf-8 -*-

import numpy as np
from random import sample
from random import choice
from scipy.ndimage.interpolation import shift
from scipy.signal import resample_poly

class data_augmentation:
        
        def __init__(self, X_train, y_train):
            self.X_train = X_train
            self.y_train = y_train
        
        def adding_noise(self, mu, sigma, iteration):    
            X_new = self.X_train
            y_new = self.y_train.copy()
            num_feature = self.X_train.shape[2]
            # num_train, num_axis, signal_length = X_train.shape
            sequence = np.array([int(i) for i in range(0,num_feature)])
            for _ in range(iteration):    
                noise =  np.random.normal(mu, sigma, self.X_train.shape)
                selection = choice(sequence)
                if selection > 0:
                    axis_without_noise = np.array(sample(list(sequence-1), selection))
                    axis_without_noise = np.sort(axis_without_noise)
                    noise[:,:,axis_without_noise] = 0
                #adding noise
                X_new = np.vstack((X_new, self.X_train + noise))
                y_new += self.y_train
            self.X_train = X_new 
            self.y_train = y_new
            return X_new, y_new
        
      
        def window_warping(self, w_start, w_end, window_ratio = [[1,2],[3,4],[4,3],[2,1]]):
            X_new = self.X_train
            y_new = self.y_train.copy() 
            w_size = w_end-w_start
            last_row = self.X_train[:,-1,:]
            for ratio in window_ratio:
                X_warp = self.X_train.copy()
                up, down = ratio[0], ratio[1]
                window = self.X_train[:,w_start:w_end,:]
                window = resample_poly(window, up, down, axis=1, padtype= 'line' )
                w_new_size = window.shape[1]
                pad = w_size - w_new_size
                X_warp[:,w_start:w_start+w_new_size,:] =  window
                if up < down: # Downsampling
                    X_pad_N = np.repeat(last_row[:, :, np.newaxis], pad, axis=2).transpose(0,2,1)
                    X_warp[:,w_start+w_new_size:-pad,:] = self.X_train[:,w_end:,:]
                    X_warp[:,-pad:,:] = X_pad_N
                else:  #  Upsampling
                    X_warp[:,w_start+w_new_size:,:] = self.X_train[:,w_end:pad,:]
                X_new = np.vstack((X_new, X_warp))
                y_new += self.y_train
            self.X_train = X_new 
            self.y_train = y_new
            return X_new, y_new
            
        
        def shift_side(self, steps = [50, 100, 150, 200]):
            X_new = self.X_train
            y_new = self.y_train.copy()
            N,h,c = self.X_train.shape
            for step in steps:
               first_row, last_row = self.X_train[:,0,:], self.X_train[:,-1,:]
               first_row = np.repeat(first_row[:, :, np.newaxis], step, axis=2).transpose(0,2,1)
               last_row = np.repeat(last_row[:, :, np.newaxis], step, axis=2).transpose(0,2,1)
               X_shift_right = shift(self.X_train,(0, step, 0), cval = 0)
               X_shift_right[: , :step, :] = first_row
               X_shift_left = shift(self.X_train,(0, -step, 0), cval = 0)
               X_shift_left[: , -step:, :] = last_row
               X_new = np.vstack((X_new, X_shift_left, X_shift_right))
               y_new += self.y_train
               y_new += self.y_train
            self.X_train = X_new 
            self.y_train = y_new        
            return X_new, y_new