# -*- coding: utf-8 -*-


from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
    
class lowpass:
    
    def __init__(self, X):
        self.X = X
    def butter_lowpass_filter(self, cutoff=2, fs=25, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, self.X, axis=1)
        return y
    
    # Filter requirements.
    
    # fs = 20.0       # sample rate, Hz
    # cutoff = 3  # desired cutoff frequency of the filter, Hz