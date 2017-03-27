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

from sklearn import ensemble, linear_model
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold




CODE_FOLDER = "/home/alex/Desktop/BNP_kaggle/"
DATA_FOLDER = "/home/alex/Desktop/BNP_kaggle/data"




def load_blend_data(DATA_FOLDER):
    import fnmatch, glob
    folder = DATA_FOLDER + '*'
    pattern = "*.p"
    l_filenames = [path for path in glob.iglob(folder) if fnmatch.fnmatch(path, pattern)]
    print(len(l_filenames), l_filenames)

    dico = pickle.load(open(l_filenames[0], 'rb'))

    test_index = dico['test_index']
    Y = dico['blend_Y']

    X = np.zeros((len(dico['blend_X']), len(l_filenames)))
    X_test = np.zeros((len(dico['blend_test_X']), len(l_filenames)))

    for i, filename in enumerate(l_filenames):
        print(filename)

        dico = pickle.load(open(filename, 'rb'))

        X[:, i] = dico['blend_X'][:, 0]
        X_test[:, i] = dico['blend_test_X']
    return X, Y, X_test, test_index


# ## Train classifiers on prediction from previous step

# #### Gathering prediction from previous layer

# In[28]:

X, Y, X_test, test_index = load_blend_data(DATA_FOLDER)
D_BLEND = (X, Y, X_test, test_index)


# In[26]:

def models():
    """
    Create a list of [DATASET, Classifier] to train on
    """
    clfs = [
        [D_BLEND, linear_model.LogisticRegression(penalty='l2') ]
        [D_BLEND, ensemble.GradientBoostingClassifier(learning_rate=0.01, n_estimators=200,max_depth=5 ) ], 
        [D_BLEND, XGBClassifier(objective="binary:logistic" ,max_depth=15, learning_rate=0.01, subsample=.8, n_estimators=800,nthread=4, seed=123)]
    ]
    return clfs


# In[27]:

clfs = models()
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
        
        # filling blend datasets with predictions
        blend_X[val_indices, 0] = y_val_pred
        blend_test_X_fold[:, fold_indice] = ytest_pred
        
        # evaluating model with logloss
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
        
    #saving relevant information
    filename = "BLENDED_{}_tr-val_{:.4f}-{:.4f}".format(clf_name, np.mean(l_train_error), np.mean(l_val_error))
    pickle.dump(dico_logs, open(DATA_FOLDER+filename + ".p" , 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    #Saving final predictions for submission
    output_filename = DATA_FOLDER + filename + '.csv'
    np.savetxt(output_filename, np.vstack((test_index, blend_test_X)).T,
               delimiter=',', fmt='%i,%.10f', header='ID,PredictedProb', comments="")




