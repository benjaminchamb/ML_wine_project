# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:53:01 2024

@author: simon
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from dtuimldmtools import rlr_validate
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection

#path = "C:\Users\simon\Desktop\Machine learning\Project 1\wine"
d = np.loadtxt("wine.data", comments="#", delimiter=",", unpack=False) # load data set 
i = 0; j = 1
attributes = [
    "Cultivars",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315diluted",
    "Proline"
]
Attri = np.array(attributes) # full name of attributes 
Attri_name = ['Cul','A', 'B' ,'C' ,'D' ,'E' ,'F', 'G', 'H', 'I', 'J', 'K', 'L' ,'M'] # Attributes sign 
cultivar = np.array(['Alpha','Beta','Gamma']) # different cultivars 
#i = 12; j = 13 # select different attributes 

## feature transformation - Standardation/normalisation
#data  = pd.DataFrame(d[:,1])
#one_hot = pd.get_dummies(data)
#df_encoded = pd.concat([data,one_hot],axis=1)
scaler = StandardScaler()
data_stand = scaler.fit_transform(d[:,1:])

    
#%
# Estimating the Generalization error is aproximatly the as Test error 

X = np.delete(d,[10],axis = 1) # parameters in winedate exept the one to be estimated 
y = d[:,10] # parameter (Color) to be estimated based on the other parameters in the wine data 
N, M = X.shape
## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.0, range(0, 6))

# Initialize variables
# T = len(lambdas)
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10

    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for the optimal value of lambda, on entire training set
    #opt_lambda
    # Changing the regularization strength the validation error has a dib before converging 
    # Also the coefficient values change start position and how fast they converge on zero
    lambdaI = opt_lambda * np.eye(M)

    
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = (
        np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test[k] = (
        np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    )
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # m = lm.LinearRegression().fit(X_train, y_train)
    # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K - 1:
        plt.figure(k, figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
        plt.xlabel("Regularization factor")
        plt.ylabel("Mean Coefficient Values")
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        # legend(attributeNames[1:], loc='best')

        plt.subplot(1, 2, 2)
        plt.title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
        plt.loglog(
            lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
        )
        plt.xlabel("Regularization factor")
        plt.ylabel("Squared error (crossvalidation)")
        plt.legend(["Train error", "Validation error"])
        plt.grid()

    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1

plt.show()
# Display results
print("Linear regression without feature selection:")
print("- Training error: {0}".format(Error_train.mean()))
print("- Test error:     {0}".format(Error_test.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()
    )
)
print("Regularized linear regression:")
print("- Training error: {0}".format(Error_train_rlr.mean()))
print("- Test error:     {0}".format(Error_test_rlr.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train_rlr.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test_rlr.sum())
        / Error_test_nofeatures.sum()
    )
)
ATT = np.delete(Attri, 10)
print("Weights in last fold:")
for m in range(M):
    print("{:>15} {:>15}".format(ATT[m], np.round(w_rlr[m, -1], 2)))


plt.loglog(
    lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
)
plt.xlabel("Regularization factor [lambda]")
plt.ylabel("Squared error (crossvalidation)")
plt.legend(["Train error", "Validation error"])
plt.grid()
#plt.savefig("estimated generalization error.pdf", format="pdf",bbox_inches="tight")


def calculate_weights(X, y, lambda_val):
    # Calculate X transpose
    X_transpose = np.transpose(X)
    
    # Calculate X transpose times X
    X_transpose_X = np.dot(X_transpose, X)
    
    # Identity matrix of appropriate size
    identity_matrix = np.identity(X_transpose_X.shape[0])
    
    # Compute regularization term
    regularization_term = lambda_val * identity_matrix
    
    # Add regularization term to X_transpose_X
    X_transpose_X_reg = X_transpose_X + regularization_term
    
    # Compute the inverse of (X_transpose_X + λI)
    X_transpose_X_reg_inv = np.linalg.inv(X_transpose_X_reg)
    
    # Compute X transpose times y
    X_transpose_y = np.dot(X_transpose, y)
    
    # Compute weights
    weights = np.dot(X_transpose_X_reg_inv, X_transpose_y)
    
    return weights

# Calculate weights

stand_X = np.delete(data_stand,[10],axis = 1)
stand_y = data_stand[:,10]

    
weights = calculate_weights(stand_X, stand_y, lambdas[1])
print("Weights Calculated:")
for m in range(M):
    print("{:>15} {:>15}".format(ATT[m], np.round(weights[m], 3)))
