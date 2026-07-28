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
from sklearn.dummy import DummyClassifier

from dtuimldmtools import train_neural_net, rlr_validate


filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine_project2_2.mat") #Use wine_project2 for X with full Data
# Load data from matlab file
mat_data = loadmat(filename)
X = mat_data['X'] # All features without Color Intensity and Cultivars
y = mat_data['y'].squeeze() # Color Intensity so what we want to predict
y_ANN = mat_data['y'].reshape(-1, 1)
X_ANN = X = X - np.ones((X.shape[0], 1)) * np.mean(X, 0)

y = y - 1
y_ANN = y_ANN - 1

attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
N, M = X.shape
# C = 2
classNames = [name[0][0] for name in mat_data["classNames"]]
C = len(classNames)


## Define the models
def linear_regression():
    return lm.LinearRegression()

def regularized_multinorm_logistic_regression(regularization_strength):
    return lm.LogisticRegression(
        solver="lbfgs",
        max_iter=10000,
        multi_class="multinomial",
        tol=1e-4,
        random_state=1,
        penalty="l2",
        C=1 / regularization_strength,
    )

# Parameters for neural network classifier
n_hidden_units = 1  # number of hidden units
n_replicates = 2  # number of networks trained in each k-fold
max_iter = 10000

# Define the model
# ANN_model = lambda: torch.nn.Sequential(
#     torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
#     torch.nn.Tanh(),  # 1st transfer function,
#     torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
#     # no final tranfer function, i.e. "linear output"
# )

loss_fn = torch.nn.CrossEntropyLoss()  # CrossEntropy since multi-class


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
best_hidden_layer_list = []

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

    X_train_outer_ANN = X_ANN[train_index_outer, :]
    y_train_outer_ANN = torch.Tensor(y_ANN[train_index_outer]) # Reshaped Tensor
    X_test_outer_ANN = X_ANN[test_index_outer, :] # Maybe need to normalize data
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

        X_train_inner_ANN = X_train_outer_ANN[train_index_inner, :] # Maybe need to normalize data
        #y_train_inner_ANN = torch.Tensor(y_train_outer_ANN_inner[train_index_inner]) # Reshaped Tensor
        X_val_ANN = X_train_outer_ANN[val_index, :] # Maybe need to normalize data
        y_val_ANN = torch.Tensor(y_train_outer_ANN_inner[val_index]) # Reshaped Tensorv


        # Train models on the Data train obtained from the outer fold 
        # Baseline. Linear Regression #
        # linear_model_inner = linear_regression()
        # linear_model_inner.fit(X_train_inner, y_train_inner)
        # y_val_pred_L = linear_model_inner.predict(X_val)
        # linear_regression_errors_val.append(np.square(y_val-y_val_pred_L).sum()/y_val.shape[0])

        ######## BASELINE ########
        # y_ave = np.mean(y_train_inner)
        # y_ave_array = np.ones(len(y_val)) * y_ave
        # linear_regression_errors_val.append(np.square(y_val-y_ave_array).sum()/y_val.shape[0])

        min_value1 = 10
        min_value2 = 10

        # Define the list of values
        values = [0.01, 0.1, 1, 10, 100]

        #####################################################################
        # Regularized Multinorm Regression find best model for current fold #
        for i in values: 
            
            #Training the inner model
            regularized_model_inner = regularized_multinorm_logistic_regression(i) #function for multi-class
            regularized_model_inner.fit(X_train_inner, y_train_inner)
            
            #Validation error
            y_val_pred_R = regularized_model_inner.predict(X_val)
            E_val_best_model = np.sum(y_val_pred_R != y_val) / len(y_val)

            # Number of miss-classifications
            # print("Error rate: \n\t {0} % out of {1}".format(E_val_best_model * 100, len(y_val)))
            
            if E_val_best_model < min_value1: 
                min_value1 = E_val_best_model
                best_lambda = i
                # print('Best_lambda=', best_lambda)


        #####################################################################
        # ANN find best model for current fold #
        for n_hidden_units in range(1, 5, 1):
            ANN_model_inner = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(n_hidden_units, C),  # n_hidden_units to 1 output neuron
                torch.nn.Softmax(dim=1),  # final tranfer function, normalisation of logit output
            )

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(
                ANN_model_inner,
                loss_fn,
                X=torch.tensor(X_train_inner_ANN, dtype=torch.float),
                y=torch.tensor(y_train_inner, dtype=torch.long),
                n_replicates=n_replicates,
                max_iter=max_iter,
            )
           
            # Determine probability of each class using trained network
            softmax_logits = net(torch.tensor(X_val_ANN, dtype=torch.float))
            # Get the estimated class as the class with highest probability (argmax on softmax_logits)
            y_val_est_ANN = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
            # Determine errors
            # e = y_test_est != y_val_ANN
            E_val_best_model_ANN = np.sum(y_val_est_ANN != y_val.squeeze()) / len(y_val)
            print(y_val)
            print(y_val_est_ANN)
            # print("Error rate: \n\t {0} % out of {1}".format(E_val_best_model_ANN * 100, len(y_val_ANN)))
            # Determine errors and errors
            # se = (y_val_est.float() - y_val_ANN.float()) ** 2  # squared error
            # mse = (sum(se).type(torch.float) / len(y_val_ANN)).data.numpy()  # mean
            # print(mse)

            if E_val_best_model_ANN < min_value2: 
                min_value2 = E_val_best_model_ANN
                best_h = n_hidden_units
                # print('Best_h=', best_h)

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
    best_lambda_list.append(best_lambda)
    print('Best ANN model for this inner fold', k+1, best_h)
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
    # y_ave = np.mean(y_train_outer)
    # y_ave_array = np.ones(len(y_test_outer)) * y_ave
    # linear_regression_errors.append(np.square(y_test_outer-y_ave_array).sum()/y_test_outer.shape[0])

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train_outer, y_train_outer)
    linear_regression_errors.append(100 * np.sum(dummy_clf.predict(X_test_outer) != y_test_outer) / len(y_test_outer))
    # Number of miss-classifications
    print(
        "Percentage of miss-classifications for baseline:\n\t {0} %".format(
            100 * np.sum(dummy_clf.predict(X_test_outer) != y_test_outer) / len(y_test_outer)
        )
    )

    # Regularized Linear Regression #
    # regularized_model = regularized_linear_regression(best_lambda)
    # regularized_model.fit(X_train_outer, y_train_outer)
    # y_test_pred_R = regularized_model.predict(X_test_outer)
    # regularized_linear_regression_errors.append(np.square(y_test_outer-y_test_pred_R).sum()/y_test_outer.shape[0])
    
    regularized_model = regularized_multinorm_logistic_regression(best_lambda) #function for multi-class
    regularized_model.fit(X_train_outer, y_train_outer)
    y_test_pred_R = regularized_model.predict(X_test_outer)
    E_test_best_model_multinorm_reguralized = 100 * np.sum(y_test_pred_R != y_test_outer) / len(y_test_outer)
    regularized_linear_regression_errors.append(E_test_best_model_multinorm_reguralized)
    
    print(
        "Percentage of miss-classifications for Regularized Multinorm Logistic Regression:\n\t {0} %".format(
            E_test_best_model_multinorm_reguralized
        )
    )
    

    # Artificial Neural Network #

    ANN_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, best_h),  # M features to n_hidden_unitsorch.nn.Linear(M, best_h)
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(best_h, C),  # n_hidden_units to 1 output neuron
        torch.nn.Softmax(dim=1),  # final tranfer function, normalisation of logit output
    )

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(
        ANN_model,
        loss_fn,
        X=torch.tensor(X_train_outer_ANN, dtype=torch.float),
        y=torch.tensor(y_train_outer.squeeze(), dtype=torch.long),
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(X_test_outer_ANN, dtype=torch.float))
    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
    y_test_est_ANN = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
    print(y_test_outer)
    print(y_test_est_ANN)
    # Determine errors
    # e = y_test_est != y_val_ANN
    E_test_best_model_ANN = 100 * np.sum(y_test_est_ANN != y_test_outer.squeeze()) / len(y_test_outer)
    neural_network_errors.append(E_test_best_model_ANN)
    print(
        "Percentage of miss-classifications for ANN:\n\t {0} %".format(
            E_test_best_model_ANN
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
    
    
    
    
    
    
    
# # %%
#     # Determine estimated class labels for test set
#     y_test_est = net(X_test_outer_ANN)

#     # Determine errors and errors
#     se = (y_test_est.float() - y_test_outer_ANN.float()) ** 2  # squared error
#     mse = (sum(se).type(torch.float) / len(y_test_outer_ANN)).data.numpy()  # mean
#     neural_network_errors.append(mse)  # store error rate for current CV fold

#     k +=1


# # Calculate mean errors across folds
# mean_linear_regression_error = np.mean(linear_regression_errors)
# mean_regularized_linear_regression_error = np.mean(regularized_linear_regression_errors)
# mean_neural_network_error = np.mean(neural_network_errors)

# print("Estimated Generalization Error for Baseline:", mean_linear_regression_error)
# print('\n')
# print("Estimated Generalization Error for Regularized Linear Regression Error:", mean_regularized_linear_regression_error)
# print('\n')
# print("Estimated Generalization Error Neural Network Error:", mean_neural_network_error)
# # Print the average classification error rate
# print(
#     "\nEstimated generalization error, RMSE: {0}".format(
#         round(np.sqrt(np.mean(neural_network_errors)), 4)
#     )
# )
