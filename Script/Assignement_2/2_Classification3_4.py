import numpy as np
import scipy.stats as st
import os
import torch
import importlib_resources
from scipy.io import loadmat
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, plot, show, xlabel, ylabel
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from dtuimldmtools import train_neural_net
from dtuimldmtools.statistics.statistics import correlated_ttest
#from Project2_question1 import *

filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine_project2_stat.mat") # See wine.mat to see how the data is set up
# Load Matlab data file and extract variables of interest

r1 = []
r2 = []
r3 = []
K = 10

# C = mat_data["C"][0, 0]
# M = mat_data["M"][0, 0]
# N = mat_data["N"][0, 0]

# attributeNames = [i[0][0] for i in mat_data["attributeNames"]]
# classNames = [j[0] for i in mat_data["classNames"] for j in i]

# Results from when I launched the 
# mA is the Baseline
mse_A_list = [61.111111111111114, 66.66666666666667, 61.111111111111114, 44.44444444444444, 50.0, 55.55555555555556, 55.55555555555556, 66.66666666666667, 64.70588235294117, 76.47058823529412]
# mB is the Regularized Model
mse_B_list = [5.555555555555555, 27.77777777777778, 11.11111111111111, 0.0, 5.555555555555555, 0.0, 5.555555555555555, 16.666666666666668, 11.764705882352942, 0.0]
# mC is the ANN
mse_C_list = [0.0, 5.555555555555555, 11.11111111111111, 0.0, 5.555555555555555, 0.0, 5.555555555555555, 0.0, 11.764705882352942, 5.882352941176471]




r1.append( np.subtract(mse_A_list, mse_B_list) ) # Baseline vs Multinom
r2.append( np.subtract(mse_A_list, mse_C_list) ) # Baseline vs ANN
r3.append( np.subtract(mse_B_list, mse_C_list) ) # Multinom vs ANN
# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K

r1 = np.transpose(r1)
r2 = np.transpose(r2)
r3 = np.transpose(r3)

p1_setupII, CI1_setupII = correlated_ttest(r1, rho, alpha=alpha)
p2_setupII, CI2_setupII = correlated_ttest(r2, rho, alpha=alpha)
p3_setupII, CI3_setupII = correlated_ttest(r3, rho, alpha=alpha)


#7.2.3
print("p-value Baseline vs Regularized Model:", p1_setupII)
print("CI Baseline vs Regularized Model:", CI1_setupII)
print("\n")
print("p-value Baseline vs ANN:", p2_setupII)
print("CI Baseline vs ANN:", CI2_setupII)
print("\n")
print("p-value Regularized Model vs ANN:", p3_setupII)
print("CI Regularized vs ANN:", CI3_setupII)




