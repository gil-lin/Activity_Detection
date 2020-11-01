# -*- coding: utf-8 -*-

import numpy as np
from random import sample
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score 
from data_augmentation import data_augmentation 
from feature_extraction import feature_extraction
import matplotlib.pyplot as plt
import seaborn as sns
from lowpass import lowpass
from sklearn.feature_selection import VarianceThreshold

def scaling(x):
    for sample in range(x.shape[0]):
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaler.fit(x[sample])
        x[sample] = (scaler.transform(x[sample]))
    return x

def highest_features(svm_coef, indices, n_features, axis, columns, index):
    svm_coef_matrix = np.reshape(svm_coef, (n_features, axis))
    svm_coef_matrix[abs(svm_coef_matrix)<abs(svm_coef[indices]).min()] = 0
    high_feature_df = pd.DataFrame(data=abs(svm_coef_matrix), index = index, columns = columns)
    # high_feature_df = high_feature_df.loc[(high_feature_df!=0).any(axis=1)]
    return high_feature_df

############################## Parameters  ####################################
classifier_name = 'cnn_1d' # svm, cnn_1d, lstm, logistic_reg, mix, polynomial, d_tree
scaling_flag = True
data_augmentation_flag = True
feature_extraction_flag = False
lowpass_flag = True
down_sampling = False
cross_val = False
d_s_factor = 2**1
feature_to_drop = ['Mag_X','Mag_Y','Mag_Z'] # 'Acc_', 'Gyr_', 'Mag_'
# parameters = [ {'C':[1, 10, 35, 200], 'kernel':['linear'], 'gamma':['scale',0.001] } ] #svm
parameters = [{'criterion':['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth':[3,4,5], 'min_samples_split':[26,20]}]
path_to_read = 'Kaggle\\clean_Chest_dataset\\ALL\\'
mu,sigma = 0, 0.5
factor = 1
counter = 0
###############################################################################
# load the dataset
y = []
for file in glob.glob(path_to_read + "*.csv"):
    filename = file.split('\\')[-1]
    if filename.split('.')[0][:3] =='ADL':
        y.append(0)
    else:
        y.append(1)
    df = pd.read_csv(file)       
    df.drop('Unnamed: 0', axis=1, inplace = True)
    for feature in feature_to_drop:
        df.drop(feature, axis=1, inplace = True)
    if df.shape[0] != 500:
        print(file)
    if counter == 0:
        X = np.array(df)
    elif counter == 1:
        X = np.array([X, np.array(df)])
    else:
        X = np.vstack((X, np.array(df)[None]))
        
    counter += 1 

indices = np.arange(X.shape[0])

# Down sampling
if down_sampling:
    X = X[:,::d_s_factor,:]
if lowpass_flag:
    lp = lowpass(X)
    X = lp.butter_lowpass_filter(cutoff=3,fs=25,order=5)

# split
X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, y, indices, test_size=0.4)

if feature_extraction_flag:
    c, K = 1, 'linear'
    # Data Augmentation 
    if data_augmentation_flag:    
        da = data_augmentation(X_train, y_train)
        X_train, y_train = da.adding_noise(mu, sigma, factor)
        X_train, y_train = da.shift_side(steps = [50,100])
        X_train, y_train = da.window_warping(200, 300, window_ratio = [[1,2],[2,1]])
    fe = feature_extraction(X_train)
    X_train, index, cross_corelation = fe.feature_extraction()
                                                       
    fe = feature_extraction(X_test)
    X_test, index, cross_corelation = fe.feature_extraction()
    
    fe = feature_extraction(X)
    X,_,__ = fe.feature_extraction()
                                                       
    #   train
    n_timesteps, n_features, axis = X_train.shape
    X_traind_2d = X_train.reshape((n_timesteps,n_features*axis))
    #   test
    n_timesteps, n_features, axis = X_test.shape
    X_test_2d = X_test.reshape((n_timesteps,n_features*axis))
    if cross_corelation is not None:
       X_traind_2d = np.concatenate((X_traind_2d,cross_corelation[idx1]), axis=1)
       X_test_2d = np.concatenate((X_test_2d,cross_corelation[idx2]), axis=1)

else:
    # Data Augmentation 
    if data_augmentation_flag:    
        da = data_augmentation(X_train, y_train)
        X_train, y_train = da.adding_noise(mu, sigma, factor)
        X_train, y_train = da.shift_side(steps = [100])
        X_train, y_train = da.window_warping(200, 300, window_ratio = [[1,2],[2,1]])
        
    if scaling_flag:
    # Min-Max Scaling (0,1)
        #train
        X_train = scaling(X_train)
        #test
        X_test = scaling(X_test)
    
# classifiers

if cross_val:
    n_timesteps, n_features, axis = X.shape
    X_2d = X.reshape((n_timesteps,n_features*axis))
    from sklearn.model_selection import GridSearchCV
    if classifier_name == 'svm':
        from sklearn.svm import SVC
        clf = SVC()
    elif classifier_name == 'logistic_reg':
        from sklearn.linear_model import LogisticRegression    
        clf = LogisticRegression()
    elif classifier_name == 'd_tree':
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        
        grid_search = GridSearchCV(estimator = clf,  
                       param_grid = parameters,
                       scoring = 'accuracy',
                       cv = 5,
                       verbose=0)
        grid_search.fit(X_2d, y)
        df_clf = pd.DataFrame(grid_search.cv_results_)
        best_scores = grid_search.best_score_
        best_params = grid_search.best_params_
        print(best_scores,best_params)
    
elif classifier_name == 'svm':
    #SVM
    from sklearn.svm import SVC    
    clf = SVC(C = c ,kernel = K, probability = True, gamma = 'scale')

elif classifier_name == 'logistic_reg':
    #Logistic Regression    
    from sklearn.linear_model import LogisticRegression    
    clf = LogisticRegression(random_state=0)

elif classifier_name == 'd_tree':
     from sklearn import tree
     clf = tree.DecisionTreeClassifier("gini","best",max_depth =3, min_samples_split=15)

elif classifier_name == 'cnn_1d':
    from classifiers.cnn_1d_classifier import cnn_1d_classifier
    clf = cnn_1d_classifier(X_train, y_train, X_test, y_test)
    classifier = clf.build()
    clf.fit(classifier,epochs=30,batch_size=64)
    
#fit
if not cross_val:
    clf = clf.fit(X_traind_2d, y_train)
    y_pred = clf.predict(X_test_2d)
    y_pred_train = clf.predict(X_traind_2d)


# Evaluation
if not cross_val:
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)     
    print('{0} Classifier accuracy:{1}'.format(classifier_name,ac))
    print('Confusion matrix {0} Classifier:{1}'.format(classifier_name,cm))
    # cm = confusion_matrix(y_train, y_pred_train)
    # ac = accuracy_score(y_train, y_pred_train)     
    # print('{0} Classifier accuracy train:{1}'.format(classifier_name,ac))
    # print('Confusion matrix {0} Classifier train:{1}'.format(classifier_name,cm))

arg_false = np.logical_xor(y_test, y_pred)
id_false= idx2[arg_false]
id_false.sort()
# #ADL
# print("ADL False: {0}".format(id_false[id_false<241]))
# print("Falls False: {0}".format(id_false[id_false>240] - 240))

#plotting
if classifier_name == 'svm':
    svm_coef = np.squeeze(clf.coef_)
    prob_svm = clf.predict_proba(X_test_2d)
    # index = ['mean','std','med', 'max', 'min', 'iqr', 'skew', 'kurt', 'amv', 'var', 'rms', 'E', 'sra', 'pp', 'cf', 'if', 'mf', 'sf']    
    indices = np.abs(svm_coef).argsort()[-40:][::-1]
    # print((indices/18).astype(int))
    # print(indices % 18)
    # svm_high_values = svm_coef[indices]       
    h_features = highest_features(svm_coef.copy(), indices, n_features, axis, df.columns, index)
    # print(h_features.index)
    
    sns.heatmap(h_features, annot=True)

elif classifier_name == "d_tree":
    plt.figure(figsize=(15,7.5))
    tree.plot_tree(clf,filled = True, rounded=True,class_names = ['ADL','Fall'])
