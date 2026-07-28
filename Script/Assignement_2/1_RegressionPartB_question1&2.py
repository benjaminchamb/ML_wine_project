# Project 2 Regression Part b
import importlib_resources
import numpy as np
import torch
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
from sklearn import model_selection

from dtuimldmtools import train_neural_net, rlr_validate

filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine_project2.mat") #Use wine_project2 for X with full Data
# Load data from matlab file
mat_data = loadmat(filename)
X = mat_data['X'] # All features without Color Intensity and Cultivars
y = mat_data['y'].squeeze() # Color Intensity so what we want to predict
y_ANN = mat_data['y'].reshape(-1, 1)
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
N, M = X.shape
C = 2

## Define the models
def linear_regression():
    return lm.LinearRegression()

def regularized_linear_regression(opt_lambda):
    return lm.Ridge(alpha=opt_lambda, fit_intercept=True)

# Parameters for neural network classifier
# n_hidden_units = 1  # number of hidden units
n_replicates = 2  # number of networks trained in each k-fold
max_iter = 10000

# Define the model
# ANN_model = lambda: torch.nn.Sequential(
#     torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
#     torch.nn.Tanh(),  # 1st transfer function,
#     torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
#     # no final tranfer function, i.e. "linear output"
# )

loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss


## Crossvalidation
# Create crossvalidation partition for evaluation
K_outer = 10
K_inner = 5
CV_outer = model_selection.KFold(n_splits=K_outer, shuffle=True)
CV_inner = model_selection.KFold(n_splits=K_inner, shuffle=True)

# Initialize Variable
#opt_lambda = 10

linear_regression_errors = [] # test error
regularized_linear_regression_errors = []
neural_network_errors = []
best_lambda_list = []
best_hidden_layer_list= []

# Normalized data
X_norm = stats.zscore(X) # Has a beneficial effect on the Linear Model and the Regularized model helping them better predict
best_h = -1
k =0
for train_index_outer, test_index_outer in CV_outer.split(X):
    print("\nOuter Crossvalidation fold: {0}/{1}".format(k + 1, K_outer))

# extract training and test set for current CV fold
    X_train_outer = X[train_index_outer,:]
    y_train_outer = y[train_index_outer]
    X_test_outer = X[test_index_outer,:]
    y_test_outer = y[test_index_outer]

    y_train_outer_ANN_inner = y_train_outer.reshape(-1, 1)

    X_train_outer_ANN = torch.Tensor(X_norm[train_index_outer, :]) # Maybe need to normalize data
    y_train_outer_ANN = torch.Tensor(y_ANN[train_index_outer]) # Reshaped Tensor
    X_test_outer_ANN = torch.Tensor(X_norm[test_index_outer, :]) # Maybe need to normalize data
    y_test_outer_ANN = torch.Tensor(y_ANN[test_index_outer]) # Reshaped Tensor

    # Initialization for Inner cross-validation
    # X_train_outer_norm = stats.zscore(X_train_outer)
    linear_regression_errors_val = [] # test error
    regularized_linear_regression_errors_val = []
    neural_network_errors_val = []

    # Inner cross-validation to see what's the best model
    X_train_outer_norm = stats.zscore(X_train_outer)
    j=0
    for train_index_inner, val_index in CV_inner.split(X_train_outer):
        print("\nInner Crossvalidation fold: {0}/{1}".format(j + 1, K_inner))
        
        X_train_inner = X_train_outer[train_index_inner,:]
        y_train_inner = y[train_index_inner]
        X_val = X_train_outer[val_index,:]
        y_val = y[val_index]

        X_train_inner_ANN = torch.Tensor(X_train_outer[train_index_inner, :]) # Maybe need to normalize data
        y_train_inner_ANN = torch.Tensor(y_train_outer_ANN_inner[train_index_inner]) # Reshaped Tensor
        X_val_ANN = torch.Tensor(X_train_outer[val_index, :]) # Maybe need to normalize data
        y_val_ANN = torch.Tensor(y_train_outer_ANN_inner[val_index]) # Reshaped Tensorv


        # Train models on the Data train obtained from the outer fold 
        # Baseline. Linear Regression #
        # linear_model_inner = linear_regression()
        # linear_model_inner.fit(X_train_inner, y_train_inner)
        # y_val_pred_L = linear_model_inner.predict(X_val)
        # linear_regression_errors_val.append(np.square(y_val-y_val_pred_L).sum()/y_val.shape[0])

        ######## BASELINE ########
        y_ave = np.mean(y_train_inner)
        y_ave_array = np.ones(len(y_val)) * y_ave
        linear_regression_errors_val.append(np.square(y_val-y_ave_array).sum()/y_val.shape[0])

        min_value1 = 10
        min_value2 = 10

        # Define the list of values
        values = [0.01, 0.1, 1, 10, 100]

        # Regularized Linear Regression find best model for current fold #
        for i in values: 
            regularized_model_inner = regularized_linear_regression(i)
            regularized_model_inner.fit(X_train_inner, y_train_inner)
            y_val_pred_R = regularized_model_inner.predict(X_val)
            E_val_best_model = np.square(y_val-y_val_pred_R).sum()/y_val.shape[0]

            if E_val_best_model < min_value1: 
                min_value1 = E_val_best_model
                best_lambda = i

        # ANN find best model for current fold #
        for n_hidden_units in range(1, 3, 1):
            ANN_model_inner = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(
                ANN_model_inner,
                loss_fn,
                X=X_train_inner_ANN,
                y=y_train_inner_ANN,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )
            y_val_est = net(X_val_ANN)
            # Determine errors and errors
            se = (y_val_est.float() - y_val_ANN.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_val_ANN)).data.numpy()  # mean

            if mse < min_value2: 
                min_value2 = mse
                best_h = n_hidden_units

        # # ANN Model
        # # Train the net on training data
        # net, final_loss, learning_curve = train_neural_net(
        #     ANN_model,
        #     loss_fn,
        #     X=X_train_inner_ANN,
        #     y=y_train_inner_ANN,
        #     n_replicates=n_replicates,
        #     max_iter=max_iter,
        # )

        # # Determine estimated class labels for test set
        # y_val_est = net(X_val_ANN)

        # # Determine errors and errors
        # se = (y_val_est.float() - y_val_ANN.float()) ** 2  # squared error
        # mse = (sum(se).type(torch.float) / len(y_val_ANN)).data.numpy()  # mean
        # neural_network_errors_val.append(mse)  # store error rate for current CV fold
    

    print('Best Regularization model for this inner fold', k+1, best_lambda)
    print('Best ANN model for this inner fold', k+1, best_h)
    best_lambda_list.append(best_lambda)
    best_hidden_layer_list.append(best_h)


    # See which one has the best model: 
    # E_val_L = np.mean(linear_regression_errors_val)
    # E_val_R = np.mean(regularized_linear_regression_errors_val)
    # E_val_ANN = np.mean(neural_network_errors_val)

    # if E_val_L <= E_val_R and E_val_L <= E_val_ANN:
    #     print('The best model for this fold', k+1, 'is the Baseline and E_val_L =', E_val_L)
    
    # if E_val_R <= E_val_L and E_val_R <= E_val_ANN:
    #     print('The best model for this fold', k+1, 'is the Regularized Model and E_val_R =', E_val_R)

    # if E_val_ANN <= E_val_L and E_val_ANN <= E_val_R:
    #     print('The best model for this fold', k+1, 'is the ANN Model and E_val_ANN =', E_val_ANN)


    ## Train models on the entire outer training fold ##
    # # Baseline. Linear Regression # Change
    # linear_model = linear_regression()
    # linear_model.fit(X_train_outer, y_train_outer)
    # y_test_pred_L = linear_model.predict(X_test_outer)
    # linear_regression_errors.append(np.square(y_test_outer-y_test_pred_L).sum()/y_test_outer.shape[0])
    ######## BASELINE ########
    y_ave = np.mean(y_train_outer)
    y_ave_array = np.ones(len(y_test_outer)) * y_ave
    linear_regression_errors.append(np.square(y_test_outer-y_ave_array).sum()/y_test_outer.shape[0])

    # Regularized Linear Regression #
    regularized_model = regularized_linear_regression(best_lambda)
    regularized_model.fit(X_train_outer, y_train_outer)
    y_test_pred_R = regularized_model.predict(X_test_outer)
    regularized_linear_regression_errors.append(np.square(y_test_outer-y_test_pred_R).sum()/y_test_outer.shape[0])
    
    # Artificial Neural Network #

    ANN_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, best_h),  # M features to n_hidden_unitsorch.nn.Linear(M, best_h)
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(best_h, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(
        ANN_model,
        loss_fn,
        X=X_train_outer_ANN,
        y=y_train_outer_ANN,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    # Determine estimated class labels for test set
    y_test_est = net(X_test_outer_ANN)

    # Determine errors and errors
    se = (y_test_est.float() - y_test_outer_ANN.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(y_test_outer_ANN)).data.numpy()  # mean
    neural_network_errors.append(mse.item())  # store error rate for current CV fold

    print("#############################\n TABLE FOR REPORT \n#############################")
    print("\nOuter Crossvalidation fold: {0}/{1}".format(k + 1, K_outer))
    print("\n")
    print("Best Lambdas:")
    print(best_lambda_list)
    print("Regularized Multinomial Log Reg Errors:")
    print(regularized_linear_regression_errors)
    print("\n \n")

        
    print("Best number of hidden layers:")
    print(best_hidden_layer_list)
    print("ANN Errors:")
    print(neural_network_errors)
    print("\n \n")
            
    print("Baseline Errors:")
    print(linear_regression_errors)
    print("#############################")

    k +=1

print(linear_regression_errors)
print(regularized_linear_regression_errors)
print(neural_network_errors)

# Calculate mean errors across folds
mean_linear_regression_error = np.mean(linear_regression_errors)
mean_regularized_linear_regression_error = np.mean(regularized_linear_regression_errors)
mean_neural_network_error = np.mean(neural_network_errors)

print("Estimated Generalization Error for Baseline:", mean_linear_regression_error)
print('\n')
print("Estimated Generalization Error for Regularized Linear Regression Error:", mean_regularized_linear_regression_error)
print('\n')
print("Estimated Generalization Error Neural Network Error:", mean_neural_network_error)
# Print the average classification error rate
print(
    "\nEstimated generalization error, RMSE: {0}".format(
        round(np.sqrt(np.mean(neural_network_errors)), 4)
    )
)
print("#############################\n TABLE FOR REPORT \n#############################")
print("\n")
print("Best Lambdas:")
print(best_lambda_list)
print("Regularized Multinomial Log Reg Errors:")
print(regularized_linear_regression_errors)
print("\n \n")

    
print("Best number of hidden layers:")
print(best_hidden_layer_list)
print("ANN Errors:")
print(neural_network_errors)
print("\n \n")
        
print("Baseline Errors:")
print(linear_regression_errors)
print("#############################")
