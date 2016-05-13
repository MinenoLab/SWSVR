SW-SVR: Sliding Window-based Support Vector Regression
======================
Implementation of our paper, "Sliding window-based support vector regression for predicting micrometeorological data."
http://www.sciencedirect.com/science/article/pii/S0957417416301786

SW-SVR is a new methodology for predicting **micrometeorological data**, such as temperature and humidity.  
This method involves a novel combination of SVR and ensemble learning.

Usage:
------
### Install Anaconda ###
https://www.continuum.io/downloads

### Install Seaborn ###
We are now requiring Seaborn library to draw graphs.

    pip install seaborn

### Run tutorial ###
    python tutorial.py

To evaluate each method, We are using AMeDAS (http://www.jma.go.jp/jma/indexe.html), large-scale  micrometeorological  data  in  Tokyo.
We have already made training/testing data as pickel file in this tutorial.py as shown as follows:

+   `Training` :
    September 1, 2011 to September 1, 2015 (for 3 years)

+   `Testing` :
    September 1, 2015 to March 1, 2016 (for 6 months)

+   `Prediction horizon` :
    6 hours


Description of parameters:
----------------
By changing parameters in tutorial.py or constructor for SlidingWindowBasedSVR class, the parameters SW-SVR uses can be changed. The summary of each parameter is shown as follows:

+   `svr_cost`, `svr_epsilon`, `svr_intercept`, `svr_itr` :
    SW-SVR involves several Linear SVR, and these parameters are used in LinerSVR of scikit-learn.
    + Linear SVR: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR

+   `kapp_gamma`, `kapp_num`, `pls_compnum` :
    SW-SVR uses kernel approximation and PLS regression for pre-processing, and these parameters are used in Kernel approximation/PLS regression of scikit-learn.
    + Kernel approximation: http://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler
    + PLS regression: http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

+   `sdc_weight`, `predict_weight` :
    These parameters are used in D-SDC, our method to extract effective data for specific data prediction.

+   `n_estimators` :
    SW-SVR builds several Linear SVRs based on this parameter.

+   `n_jobs` :
    This parameter means the number of CPU used for training SW-SVR.
    if `n_jobs < 0`, all CPU are used.