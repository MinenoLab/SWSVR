'''
Created on 2015/10/05

@author: Kaneda
'''

import numpy as np
import pandas as pd

class FeaturesData:
    '''
    This is features set. This is implemented by numpy.
    '''

    def __init__(self, X, time=[]):
        '''
        Constructor
        '''
        self.X = X #2dim array
        self.time = time #1dim array

class LabeledData:
    '''
    This is training data for supervised learning.
    This is FeaturesData with label.
    '''

    def __init__(self, X, y, time=[]):
        '''
        Constructor
        '''
        self.X = X #2dim array
        self.y = y #1dim array
        self.time = time #1dim array

class SdcData:
    '''
    This is data for SDC. SDC needs two LabeledData, previous data and following data.
    e.g. in 6 hours later prediction, the SDC data has 13:00 data and 19:00 data.
    '''

    def __init__(self, pre_X, pre_y, fol_X):
        '''
        Constructor
        '''
        self.pre_X = pre_X #2dim array
        self.pre_y = pre_y #1dim array
        self.fol_X = fol_X #2dim array

    def extend(self, sdc_data):
        self.pre_X = np.append(self.pre_X, sdc_data.pre_X, axis=0)
        self.pre_y = np.append(self.pre_y,sdc_data.pre_y)
        self.fol_X = np.append(self.fol_X,sdc_data.fol_X, axis=0)

