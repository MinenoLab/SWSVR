'''
Created on 2015/10/05

@author: Kaneda
'''

from swsvr.SlidingWindowBasedSVR import SlidingWindowBasedSVR
import swsvr.Utility as ut
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import cPickle as pickle
import time

def visualization(true_y, swsvr_result, other):
    name=[]
    mape=[]
    print "+ Each prediction error for MAPE"
    name.append("swsvr")
    print "[SWSVR]",
    tmp_mape = ut.metrics(true_y, swsvr_result, is_print = True)
    mape.append(tmp_mape)

    for l in simple_learners:
        name.append(l.name)
        print "[%s]" % l.name,
        tmp_mape = ut.metrics(true_y, l.result,  is_print = True)
        mape.append(tmp_mape)

    df = pd.DataFrame({'true':true_y, 'swsvr':swsvr_result})
    for l in other:
        tmp = pd.DataFrame({l.name:l.result})
        df = pd.concat([df,tmp],axis=1)

    plt.figure(1)
    plt.subplot2grid((2, 1), (0, 0))
    sea.barplot(x=name,y=mape)
    plt.axhline(y=mape[0], color='r', ls=':')
    plt.title("MAPE")
    plt.subplot2grid((2, 1), (1, 0))
    plt.plot(true_y,lw=3,label='true')
    plt.plot(swsvr_result,label='swsvr')
    for l in other:
        plt.plot(l.result,'--',label=l.name)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='center', borderaxespad=0, ncol=10)
    plt.title("Regression curve")
    plt.show()

if __name__ == '__main__':
    is_tune=True

    # sample parameters to tune
    swsvr_params = {'svr_cost': [4, 8, 16, 32, 64, 126],
                    'svr_epsilon': [0.001, 0.00001],
                    'svr_intercept': [32, 8, 64, 16, 128],
                    'svr_itr':[100000],
                    'kapp_gamma': [0.00001],
                    'kapp_num': [100],
                    'pls_compnum': [50],
                    'sdc_weight': [0.5],
                    'predict_weight': [3],
                    'lower_limit': [10],
                    'n_estimators':[100, 500]}

    # load pkl
    with open('sdc_data_20110901_20140901_6h.pkl', 'rb') as f:
        sdc_data = pickle.load(f)
    with open('test_data_20140901_20150301_6h.pkl', 'rb') as f:
        test_data = pickle.load(f)

    if is_tune:
        swsvr = ut.grid_tuning(swsvr_params, sdc_data, test_data)
    else:
        swsvr = SlidingWindowBasedSVR()
    start = time.time()
    swsvr.train(sdc_data)
    print "+ Each building time"
    print "[SWSVR] built in %f sec" % (time.time()-start)
    simple_learners = ut.build_other_learners(sdc_data.pre_X, sdc_data.pre_y)

    # predict
    swsvr_result, true_y = swsvr.predict(test_data.X), test_data.y
    for l in simple_learners: l.predict(test_data.X)

    # visualize
    visualization(true_y, swsvr_result, simple_learners)
