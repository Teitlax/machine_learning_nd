# -*- coding: utf-8 -*-
"""
Created on Sun Jan 08 14:35:17 2017

@author: Alex
"""
__author__ = 'Alex'

import numpy as np
import pandas as pd
import pickle


from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold


CODE_FOLDER = "/home/alex/Desktop/BNP_kaggle/"
DATA_FOLDER = "/home/alex/Desktop/BNP_kaggle/data"


def load_data(filename):

    data_name = filename.split('_')[0]
    pd_data = pd.read_hdf(CODE_FOLDER + "data/" + filename)


    pd_train = pd_data[pd_data.target >= 0]
    pd_test = pd_data[pd_data.target == -1]

    Y = pd_train['target'].values.astype(int)
    test_index = pd_test['ID'].values.astype(int)

    X = np.array(pd_train.drop(['ID', 'target'],1))
    X_test = np.array(pd_test.drop(['ID','target'], 1))

    return X, Y, X_test, test_index, pd_data, data_name


#loading 4 datasets
    
    
X, Y, X_test, test_index, pd_data, data_name = load_data('D1_[LE-cat]_[NAmean].p')
X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D1 = (X, Y, X_test,test_index, data_name)


X, Y, X_test, test_index, pd_data, data_name = load_data('D2_[LE-cat]_[NA-999].p')
X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
D2 = (X, Y, X_test,test_index, data_name)

#X, Y, X_test, test_index, pd_data, data_name = load_data('D5_[OnlyCont]_[NAmean].p')
#X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
#D5 = (X, Y, X_test,test_index, data_name)
#
#X, Y, X_test, test_index, pd_data, data_name = load_data('D6_[OnlyCatLE].p')
#X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
#D6 = (X, Y, X_test,test_index, data_name)




def list_of_models():
    """
    Create a list of [DATASET, Classifier] to train on
    """
    
    
    ET_params = {'n_estimators':500,'max_features': 50,'criterion': 'entropy',
                 'min_samples_split': 4,'max_depth': 35, 'min_samples_leaf': 2}
    
    
    clfs = [
            [D2, XGBClassifier(objective="binary:logistic" ,max_depth=20, learning_rate=0.05, subsample=.8, n_estimators=500,nthread=4, seed=123)],
            [D2, XGBClassifier(objective="binary:logistic" ,max_depth=15, learning_rate=0.01, subsample=.8, n_estimators=500,nthread=4, seed=123)],
            [D1, XGBClassifier(objective="binary:logistic" ,max_depth=20, learning_rate=0.05, subsample=.8, n_estimators=500,nthread=4, seed=123)],
            [D1, XGBClassifier(objective="binary:logistic" ,max_depth=15, learning_rate=0.01, subsample=.8, n_estimators=500,nthread=4, seed=123)],
#            [D5, XGBClassifier(objective="binary:logistic" ,max_depth=10, learning_rate=0.01, subsample=.8, n_estimators=500,nthread=4, seed=123)],
#            [D6, XGBClassifier(objective="binary:logistic" ,max_depth=20, learning_rate=0.01, subsample=.8, n_estimators=500,nthread=4, seed=123)],
            [D1, ensemble.RandomForestClassifier(n_jobs=4, **ET_params)],
            [D2, ensemble.RandomForestClassifier(n_jobs=4, **ET_params)],
            [D1, ensemble.ExtraTreesClassifier(n_jobs=4, **ET_params)],
            [D2, ensemble.ExtraTreesClassifier(n_jobs=4, **ET_params)],
#            [D5, ensemble.ExtraTreesClassifier(n_jobs=4, **ET_params)],
#            [D6, ensemble.RandomForestClassifier(n_jobs=4, **ET_params)]
    ]
    return clfs
    
clfs = list_of_models()
skf = StratifiedKFold(Y, n_folds=5, shuffle=True , random_state=123)


#Cross validation from a list of models
for clf_indice, data_clf in enumerate(clfs):
    
    #Selecting a model from the list
    print("Classifier [%i]" % clf_indice)
    
    X = data_clf[0][0]
    Y = data_clf[0][1]
    X_test = data_clf[0][2]
    test_index = data_clf[0][3]
    
    clf = data_clf[1]
    clf_name = clf.__class__.__name__
    print(clf)
    
    blend_X = np.zeros((len(X), 1))
    blend_Y = Y
    blend_test_X = np.zeros((len(X_test), 1))
    blend_test_X_fold = np.zeros((len(X_test), len(skf)))
    
    l_train_error = []
    l_val_error = []
    for fold_indice, (train_indices, val_indices) in enumerate(skf):
        
        print("Fold [%i]" % fold_indice)
        x_train = X[train_indices]
        y_train = Y[train_indices]
        x_val = X[val_indices]
        y_val = Y[val_indices]
        
        clf.fit(x_train, y_train)
        
        y_train_pred = clf.predict_proba(x_train)[:,1]
        y_val_pred = clf.predict_proba(x_val)[:,1]
        ytest_pred = clf.predict_proba(X_test)[:,1]
        
        # filling blend data sets
        blend_X[val_indices, 0] = y_val_pred
        blend_test_X_fold[:, fold_indice] = ytest_pred
        
        # evaluating model
        train_error = log_loss(y_train, y_train_pred)
        val_error = log_loss(y_val, y_val_pred)
        l_train_error.append(train_error)
        l_val_error.append(val_error)
        print("train/val error: [{0:.4f}|{1:.4f}]".format(train_error, val_error))
        
        
    blend_test_X = np.mean(blend_test_X_fold, axis=1)
    
    dico_logs = {'blend_X': blend_X,
               'blend_Y': Y,
               'blend_test_X': blend_test_X,
               'test_index': test_index,
               'clf_name': clf_name}
        
    #saving relevant information for blending later
    filename = "BLEND_{}_tr-val_{:.4f}-{:.4f}".format(clf_name, np.mean(l_train_error), np.mean(l_val_error))
    pickle.dump(dico_logs, open(DATA_FOLDER+filename + ".p" , 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    #Saving final predictions
    output_filename = DATA_FOLDER + filename + '.csv'
    np.savetxt(output_filename, np.vstack((test_index, blend_test_X)).T,
               delimiter=',', fmt='%i,%.10f', header='ID,PredictedProb', comments="")




