# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:37:58 2020
@author: eyaraz
"""


#CNN 1D Classifier
#info: 


import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras import optimizers
# from utils.utils import save_logs
# from utils.utils import calculate_metrics
# from utils.utils import save_test_duration


class cnn_1d_classifier:
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
                

    def build(self, filters=[48,24], dense_units=[40,30], dropout = 0.65):
        cnn_layers, dense_layers=len(filters), len(dense_units)
        n_timesteps, n_features, n_outputs = self.X_train.shape[1], self.X_train.shape[2], self.y_train[1]
        optimizer = [optimizers.Adam(lr=0.001, decay = 5*1e-5), optimizers.RMSprop(lr=0.001, rho = 0.99, decay = 1e-4)]
        classifier = Sequential()
        for i in range(cnn_layers):
            if i == 0:
                classifier.add(Conv1D(filters=filters[i], kernel_size=4, activation='relu', input_shape=(n_timesteps,n_features)))
                classifier.add(Dropout(dropout/3))
            else:
                classifier.add(Conv1D(filters=filters[i], kernel_size=4, activation='relu'))
                classifier.add(Dropout(dropout))            
                classifier.add(MaxPooling1D(pool_size=2))
        
        classifier.add(Flatten())
        for i in range(dense_layers):
            classifier.add(Dense(dense_units[i], activation='relu'))
            classifier.add(Dropout(dropout))
    
        classifier.add(Dense(1, activation='sigmoid'))
        classifier.compile(loss='binary_crossentropy', optimizer=optimizer[0], metrics=['accuracy'])
        print(classifier.summary())
        return classifier
        

    def fit(self, classifier, epochs=30, batch_size=64):
        history = classifier.fit(self.X_train, self.y_train, epochs = epochs, batch_size = batch_size, validation_data=(self.X_test, self.y_test))
        _, accuracy = classifier.evaluate(self.X_test, self.y_test, batch_size=64, verbose=0)
        print('1_D CNN Classifier accuracy:{0}'.format(accuracy))
        if accuracy >0.95:
            classifier.save('cnn_1d_clf_fall_detector.h5')
    
        # # Plotting Performance
        import matplotlib.pyplot as plt
        
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        