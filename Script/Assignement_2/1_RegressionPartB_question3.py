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
from Project2_question1 import *

filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine_project2_stat.mat") # See wine.mat to see how the data is set up
# Load Matlab data file and extract variables of interest

# C = mat_data["C"][0, 0]
# M = mat_data["M"][0, 0]
# N = mat_data["N"][0, 0]

# attributeNames = [i[0][0] for i in mat_data["attributeNames"]]
# classNames = [j[0] for i in mat_data["classNames"] for j in i]

############### Results from when I launched the code for the report to keep the same values #######################
# # mA is the Baseline
# mse_A_list = [3.643449501847831, 4.801860767969271, 7.366877796086197, 6.094980004143403, 2.9110735379278636, 5.184120616861459, 
#               6.076485427374913, 2.690773579945139, 5.661905525933323, 9.619929630699446]
# # mB is the Regularized Model
# mse_B_list = [1.2344893894404236, 3.7537235687754884, 2.985948220163008, 1.7947260626026287, 1.8807601937228025, 2.19570644360983, 
#               3.0478343210740615, 1.3069633145590207, 2.100154745518455, 6.4423460205967755]
# # mC is the ANN
# mse_C_list = [0.9655330181121826, 1.9662041664123535, 2.8458023071289062, 2.9225640296936035, 1.9519776105880737, 1.6422431468963623, 
#               2.4132206439971924, 2.993563175201416, 3.137381076812744, 2.8292837142944336]

#####################################################################

mse_A_list = linear_regression_errors
mse_B_list = regularized_linear_regression_errors
mse_C_list = neural_network_errors

# perform statistical comparison of the models
# compute z with squared error.
zA = np.array(mse_A_list) # Baseline Model
zB = np.array(mse_B_list) # Regularized Model
zC = np.array(mse_C_list) # ANN Model

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(
    1 - alpha, df=len(zA) - 1, loc=np.mean(zA), scale=st.sem(zA)
)  # Confidence interval

# compute confidence interval of model B
alpha = 0.05
CIB = st.t.interval(
    1 - alpha, df=len(zB) - 1, loc=np.mean(zB), scale=st.sem(zB)
)  # Confidence interval

# compute confidence interval of model B
alpha = 0.05
CIC = st.t.interval(
    1 - alpha, df=len(zC) - 1, loc=np.mean(zC), scale=st.sem(zC)
)  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z_Baseline_vs_R = zA - zB
z_Baseline_vs_ANN = zA - zC
z_R_vs_ANN = zB - zC

################### Basline vs Regularized ################################
CI_Baseline_vs_R = st.t.interval(
    1 - alpha, len(z_Baseline_vs_R) - 1, loc=np.mean(z_Baseline_vs_R), scale=st.sem(z_Baseline_vs_R)
)  # Confidence interval

p_Baseline_vs_R = 2 * st.t.cdf(-np.abs(np.mean(z_Baseline_vs_R)) / st.sem(z_Baseline_vs_R), df=len(z_Baseline_vs_R) - 1)  # p-value


################### Basline vs ANN ################################
CI_Baseline_vs_ANN = st.t.interval(
    1 - alpha, len(z_Baseline_vs_ANN) - 1, loc=np.mean(z_Baseline_vs_ANN), scale=st.sem(z_Baseline_vs_ANN)
)  # Confidence interval

p_Baseline_vs_ANN = 2 * st.t.cdf(-np.abs(np.mean(z_Baseline_vs_ANN)) / st.sem(z_Baseline_vs_ANN), df=len(z_Baseline_vs_ANN) - 1)  # p-value

################### Regularized vs ANN ################################
CI_R_vs_ANN = st.t.interval(
    1 - alpha, len(z_R_vs_ANN) - 1, loc=np.mean(z_R_vs_ANN), scale=st.sem(z_R_vs_ANN)
)  # Confidence interval

p_R_vs_ANN = 2 * st.t.cdf(-np.abs(np.mean(z_R_vs_ANN)) / st.sem(z_R_vs_ANN), df=len(z_R_vs_ANN) - 1)  # p-value


#7.2.1
print("CI for model A:", CIA)
print("CI for model B:", CIB)
print("CI for model C:", CIC)
print("\n")
#7.2.3
print("p-value Baseline vs Regularized Model:", p_Baseline_vs_R)
print("CI Baseline vs Regularized Model:", CI_Baseline_vs_R)
print("\n")
print("p-value Baseline vs ANN:", p_Baseline_vs_ANN)
print("CI Baseline vs ANN:", CI_Baseline_vs_ANN)
print("\n")
print("p-value Regularized Model vs ANN:", p_R_vs_ANN)
print("CI Regularized vs ANN:", CI_R_vs_ANN)