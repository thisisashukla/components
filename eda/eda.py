import os
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


class Explore():

    def __init__(self, df):
        self.df = df
        self.num_descr = None
        self.cat_descr = None

    def numDescribe(self):
        descr = self.df.describe().T
        descr.loc[:, 'na%'] = (self.naCount(descr.index)/descr['count'])*100
        descr.loc[:, 'unique'] = [len(self.df[x].unique()) for x in descr.index]
        descr.loc[descr['count'] == 0, 'na%'] = np.nan

        cols = ['count', 'na%', 'unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        self.num_descr = descr[cols].T
        self.num_cols = self.num_descr.columns.tolist()

        return self.num_descr

    def catDescribe(self):
        try:
            assert self.num_descr != None
        except:
            print('Warning: numDescribe function needs to be called before catDescribe')
            self.numDescribe()
        
        self.cat_cols = [x for x in self.df.columns if x not in self.num_cols]
        descr = pd.DataFrame(columns = ['count', 'na%', 'unique'], index = self.cat_cols)
        descr.loc[:, 'count'] = self.df[self.cat_cols].count()
        descr.loc[:, 'na%'] = (self.naCount(descr.index)/descr['count'])*100
        descr.loc[:, 'unique'] = [len(self.df[x].unique()) for x in descr.index]
        self.cat_descr = descr.T

        return self.cat_descr

    def naCount(self, cols):
        return self.df[cols].isna().sum(axis = 0)
    
    



    