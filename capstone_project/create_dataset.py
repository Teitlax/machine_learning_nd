# -*- coding: utf-8 -*-
"""
Created on Sun Jan 08 15:41:55 2017

@author: Alex
"""

__author__ = 'Alex'


CODE_FOLDER = "/home/alex/Desktop/BNP_kaggle/"

import os
if os.getcwd() != CODE_FOLDER: os.chdir(CODE_FOLDER)
import pandas as pd
import numpy as np



class dummy_col_binss():

    def __init__(self, cols=None, prefix='LOL_', nb_bins=10):

        self.prefix=prefix
        self.nb_bins = nb_bins
        self.cols = cols
        self.bins = None

    def fit(self, data):

        self.bins = np.linspace(data[self.cols].min(), data[self.cols].max(), self.nb_bins)

        return self

    def transform(self, data):

        pd_dummy = pd.get_dummies(np.digitize(data[self.cols], self.bins), prefix=self.prefix)

        return pd_dummy



class dummy_col():


    def __init__(self, cols=None, prefix='LOL_', nb_features=10):

        self.selected_features = None
        self.rejected_features = None
        self.prefix=prefix
        self.nb_features = nb_features
        self.cols = cols

    def fit(self, pd_train, pd_test):

        #Frequent item ==> Dummify
        selected_features_train = pd_train[self.cols].value_counts().index[:self.nb_features]
        selected_features_test = pd_test[self.cols].value_counts().index[:self.nb_features]

        self.selected_features = list(set(selected_features_train).intersection(set(selected_features_test)))

        #Rare items ==> gather all into a "garbage" column
        rejected_features_train = pd_train[self.cols].value_counts().index[self.nb_features:]
        rejected_features_test = pd_test[self.cols].value_counts().index[self.nb_features:]

        self.rejected_features = list(set(rejected_features_train).intersection(set(rejected_features_test)))

    def transform(self, data):

        df_dummy = data[self.cols].apply(lambda r: r if r in self.selected_features else 'LowFreqFeat')

        #Dummy all items
        df_dummy = pd.get_dummies(df_dummy).groupby(df_dummy.index).sum()


        df_dummy = df_dummy.rename(columns=lambda x: self.prefix + str(x))

        return df_dummy

#loading datasets

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = train['target']

test['target'] = -1

df_all = train.append(test).reset_index(drop=True)
df_all = df_all.drop(['v107'],1)


cat_var = list(df_all.select_dtypes(["object"]).columns)
cont_var = list(df_all.select_dtypes(["float", "int"]).columns)
#cont_var.remove('ID')
#cont_var.remove('target')

df_all['nb_na_cont'] = df_all[cont_var].isnull().sum(1)
df_all['nb_na_cat'] = df_all[cat_var].isnull().sum(1)


#function to easily transform dataset


def transform_data(data, LECAT=False, NAMEAN=False, NA999=False, OH=False, ONLYCONT=False, ONLYCAT=False, ONLYCATOH=False, COLSREMOVAL=False, cols=[], maxCategories=300):

    data = data.copy()

    cat_var = list(data.select_dtypes(["object"]).columns)
    cont_var = list(data.select_dtypes(["float", "int"]).columns)

    if COLSREMOVAL:
        data = data.drop(cols, 1, inplace=False)
        cat_var = list(data.select_dtypes(["object"]).columns)
        cont_var = list(data.select_dtypes(["float", "int"]).columns)


    if NAMEAN:
        for col in cont_var:
            data.loc[data[col].isnull(), col] = data[col].mean()

    if NA999:
        for col in cont_var:
            data.loc[data[col].isnull(), col] = -999

    if LECAT:
        for col in data[cat_var]: data[col] = pd.factorize(data[col])[0]

    if OH:
        cols2dummy = [col for col in data[cat_var] if len(data[col].unique()) <= maxCategories]
        colsNot2dummy = [col for col in data[cat_var] if len(data[col].unique()) > maxCategories]
        data = pd.get_dummies(data, dummy_na=True, columns=cols2dummy)

        #binning
        for col in colsNot2dummy:
            data[col] = pd.factorize(data[col])[0]
            dcb = dummy_col_binss(cols=col, prefix=col, nb_bins=2000)
            dcb.fit(data)
            pd_binned = dcb.transform(data)
            data = pd.concat([data,pd_binned],1)
    if ONLYCONT:
        data = data[cont_var]

    if ONLYCAT:
        test_idx = data['ID']
        Y = data['target']
        data = data[cat_var]
        data['ID'] = test_idx
        data['target'] = Y

    if ONLYCATOH:
        test_idx = data['ID']
        Y = data['target']
        cols = list(set(data.columns).difference(set(cont_var))) ; print(cols)
        data = data[cols]
        data['ID'] = test_idx
        data['target'] = Y


    return data


D1 = transform_data(data=df_all, LECAT=True, NAMEAN=True)
D1.to_hdf(CODE_FOLDER + '/data/D1_[LE-cat]_[NAmean].p', 'wb')

D2 = transform_data(data=df_all, LECAT=True, NA999=True)
D2.to_hdf(CODE_FOLDER + '/data/D2_[LE-cat]_[NA-999].p', 'wb')

#D3 = transform_data(data=df_all, OH=True, NAMEAN=True)
#D3.to_hdf(CODE_FOLDER + '/data/D3_[OH300]_[NAmean].p', 'wb')


#D4 = transform_data(data=df_all, OH=True, NA999=True)
#D4.to_hdf(CODE_FOLDER + '/data/D4_[OH300]_[NA-999].p', 'wb')

D5 = transform_data(data=df_all, ONLYCONT=True, NAMEAN=True)
D5.to_hdf(CODE_FOLDER + '/data/D5_[OnlyCont]_[NAmean].p', 'wb')


D6 = transform_data(data=df_all, ONLYCAT=True, LECAT=True)
D6.to_hdf(CODE_FOLDER + '/data/D6_[OnlyCatLE].p', 'wb')

#D7 = transform_data(data=df_all, ONLYCATOH=True, OH=True)
#D7.to_hdf(CODE_FOLDER + '/data/D7_[OnlyCatOH].p', 'wb')



















