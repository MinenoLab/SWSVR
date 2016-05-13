'''
Created on 2015/10/07

@author: Kaneda
'''
import sys
import datetime
import math
import locale
import time
import datetime
import Converter
import time
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.grid_search import ParameterGrid
from numba.decorators import jit
from sklearn.svm import SVR
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from multiprocessing import Pool, cpu_count, current_process
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from multiprocessing import Pool, cpu_count, current_process
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
from sklearn.gaussian_process import GaussianProcess
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

#################################
# methods related other methods #
#################################

class SimpleLearner:
    def __init__(self,name, model):
        '''
        Constructor
        '''
        self.name = name
        self.model = model
        self.result = []
        self.time = 0
        self.scaler = StandardScaler()

    def fit(self, X, y):
        start = time.clock()
        self.model.fit(X,y)
        self.time = time.clock() - start

    def predict(self, X):
        if self.name == "linearSVR" or self.name == "pa":
            self.result = np.array(self.model.predict(self.scaler.transform(X)))
        else:
            self.result = np.array(self.model.predict(X))

class PersistenceModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self,X, y):
        return self

    def predict(self,X):
        return X[:,1]

def build_other_learners(train_x, train_y):
    simple_learners = []
    simple_learners.append(SimpleLearner("rf",RandomForestRegressor(n_jobs=-1, max_features=0.6, n_estimators=2, max_depth=8)))
    simple_learners.append(SimpleLearner("gb",GradientBoostingRegressor(n_estimators=10, loss='huber', learning_rate=0.5, max_depth=4)))
    simple_learners.append(SimpleLearner("linearSVR",LinearSVR(intercept_scaling=64, C=128, max_iter=1000, dual=False, loss='squared_epsilon_insensitive')))
    #simple_learners.append(SimpleLearner("svr",SVR(C=100, epsilon=0.001, gamma=0.00001)))
    for sl in simple_learners:
        start = time.time()
        if sl.name == "linearSVR" or sl.name == "pa":
            sl.scaler.fit(train_x)
            s_train_x = sl.scaler.transform(train_x)
            sl.fit(s_train_x,train_y)
        else:
            sl.fit(train_x,train_y)
        print "[%s] built in %f sec" % (sl.name,(time.time()-start))
    return simple_learners

##########################
# method related metrics #
##########################

def metrics(y_true, y_pred, is_print = False):
    y_pred = np.delete(y_pred, np.where(y_true==0), 0)
    y_true = np.delete(y_true, np.where(y_true==0), 0)
    tmp_mape = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(tmp_mape) * 100
    if is_print:
        print "MAPE:",mape
    return mape

##############################
# method related calculation #
##############################

@jit('f8(f8[:],f8[:])')
def cal_weighted_average(vector, weight_vector):
    return np.average(vector,weights=weight_vector)

@jit('f8(f8[:],f8[:])')
def calc_euclid(A, B):
    '''
    Args:
        features: numpy
    '''
    return np.linalg.norm(A - B)

@jit('f8[:](f8[:],f8)')
def calc_reciprocal(double, weight_param):
    '''
    This method calc weight from a double value.
    '''
    denominator = np.power(double, weight_param)
    index = np.where(denominator == 0)[0]
    for i in index:# double(arg1) has some zero
        denominator[i] = np.min(denominator[np.where(denominator>0)])
    denominator = np.reciprocal(denominator)

    return denominator

@jit('f8[:](f8[:,:],f8[:,:])')
def calc_euclid_matrix(from_data, to_data):
    '''
    This method calculates euclid distances for matrix.
    Parameters must be FeaturesData class.
    '''
    return np.linalg.norm((from_data - to_data), axis=1)

###############################
# method related SWSVR tuning #
###############################

def grid_tuning(params, sdc_data, test_data):
    from swsvr.SlidingWindowBasedSVR import SlidingWindowBasedSVR

    params_grid = list(ParameterGrid(params))
    best_mae = 10000
    best_rmse = 10000
    best_mape = 10000
    print "\nIndex, Time, MAPE(min), MAPE(ave), MAPE(std),",
    for k in params_grid[0].keys():
        print k+",",
    print ""
    swsvr = SlidingWindowBasedSVR()
    for (index, params) in enumerate(params_grid):
        tmp_mape_list=[]
        progress=0
        for i in range(1):
            swsvr.init(svr_cost=params["svr_cost"], svr_epsilon=params["svr_epsilon"],svr_intercept=params["svr_intercept"],svr_itr=params["svr_itr"],
                                kapp_gamma=params["kapp_gamma"], kapp_num=params["kapp_num"],
                                pls_compnum=params["pls_compnum"],
                                sdc_weight=params["sdc_weight"], predict_weight=params["predict_weight"],
                                lower_limit=params["lower_limit"],
                                n_estimators=params["n_estimators"])
            start = time.clock()
            swsvr.train(sdc_data)
            progress = time.clock() - start
            swsvr_result = swsvr.predict(test_data.X)
            tmp_mape = metrics(test_data.y, swsvr_result)
            tmp_mape_list.append(tmp_mape)
        tmp_mape_list = np.array(tmp_mape_list)
        condition = np.where(tmp_mape_list == np.min(tmp_mape_list))[0]
        print (str(index)
               +", "+str(progress)
               +", "+str(np.min(tmp_mape_list))
               +", "+str(np.average(tmp_mape_list))
               +", "+str(np.std(tmp_mape_list))
               +", "),
        for v in params.values():
            print str(v)+",",
        print ""
        sys.stdout.flush()
        if best_mape > np.min(tmp_mape_list):
            best_params = params
            best_mape = tmp_mape_list[condition][0]
    #swsvr._pool_close() #when swsvr run with single process, error appear
    #Tune is finished
    print "[TUNE RESULT] MAPE:"+str(best_mape)+"::"+ str(best_params)
    best_swsvr = SlidingWindowBasedSVR(svr_cost=best_params["svr_cost"], svr_epsilon=best_params["svr_epsilon"],svr_intercept=best_params["svr_intercept"],svr_itr=best_params["svr_itr"],
                             kapp_gamma=best_params["kapp_gamma"], kapp_num=best_params["kapp_num"],
                             pls_compnum=params["pls_compnum"],
                             sdc_weight=best_params["sdc_weight"], predict_weight=best_params["predict_weight"],
                             lower_limit=best_params["lower_limit"],
                             n_estimators=params["n_estimators"])
    return best_swsvr