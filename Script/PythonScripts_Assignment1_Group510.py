# First Python Script used for doing some of the plots. Second script at line 147!

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy import stats
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

y = d[:,0] # array of cultivars 
M = len(attributes)
N = len(d[:,0])
Stat_Means = np.zeros((14,1)) # empty array
Stat_StdDev = np.zeros((14,1))
for i in range(0,14):    # calc the std and mean for each attribute with for loop
    Stat_Means[i] = np.mean(d[:,i])
    Stat_StdDev[i] = np.std(d[:,i])

norm1_d = normalize(d, axis=1, norm='l1')

norm_d = np.zeros((178,14))
for i in range(1,14):
    norm_d[:,i] = (d[:,i]- np.transpose(Stat_Means[i]))/Stat_StdDev[i]


#%
# Box plot
#plt.boxplot(d)
#plt.xticks(range(1, 15), attributes, rotation=45)
#plt.tight_layout()
#plt.savefig('Boxdata.pdf')
#plt.show()

#%
# Histogram 
# x = np.zeros((178,14))
# nbins = 20
# pdf = np.zeros((178,14))
# for i in range(1,14):
#     x[:,i] = np.linspace(d[:,i].min(), d[:,i].max(), 178)
#     pdf[:,i] = stats.norm.pdf(d[:,i], loc=Stat_Means[i], scale=Stat_StdDev[i])

# plt.figure(figsize=(12, 7))
# u = np.floor(np.sqrt(M))
# v = np.ceil(float(M) / u)
# for i in range(M):
#     plt.subplot(int(u), int(v), i+1)
#     plt.hist(d[:, i],bins=nbins,density=True)
#     plt.xlabel(attributes[i])
#     plt.ylim(0, 3.5)  # Make the y-axes equal for improved readabilit
#     plt.plot(d[:,i], pdf[:,i], ".", color="red")
#     if i % v != 0:
#         plt.yticks([])
#     if i == 0:
#         plt.title("Wine: Histogram")
# plt.tight_layout()
#plt.savefig('Histo_data.pdf')
#plt.show()

#%

# # plot attribute against each other
# f = plt.figure()
# plt.title("Winedata") #plot title 

# for c in range(1,len(cultivar)+1):
#     # select indices belonging to class c:
#     class_mask = y == c # selecting indices belonging to the different cultivars 
#     plt.plot(d[class_mask, i], d[class_mask, j], "o", alpha=0.3) # ploting attributes and cultivars

# plt.legend(cultivar) # legeng with names
# plt.xlabel(Attri[i]) # x label attribute
# plt.ylabel(Attri[j]) # y label attribute
# #plt.savefig('look.pdf')
# #plt.show()

# %
from scipy.linalg import svd
# Subtract mean value from data
X = np.delete(d,0,1)
Y = X - np.ones((N, 1))*X.mean(axis=0)
Y = Y*1/(X.std(axis=0))
# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)
# Compute variance explained by principal components
V = V.T
Z = Y@V

rho = (S * S) / (S * S).sum()
threshold = 0.9


i = 3
j = 0
# Plot PCA of the data
f = plt.figure()
plt.title("Wine data: PCA")
# Z = array(Z)
for c in range(1,len(attributes)+1):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
plt.legend(cultivar)
plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))

# Output result to screen
plt.savefig('PCA4.pdf')
plt.show()


# # Plot variance explained
# plt.figure()
# plt.plot(range(1, len(rho) + 1), rho, "x-")
# plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
# plt.plot([1, len(rho)], [threshold, threshold], "k--")
# plt.title("Variance explained by principal components")
# plt.xlabel("Principal component")
# plt.ylabel("Variance explained")
# plt.legend(["Individual", "Cumulative", "Threshold"])
# plt.grid()
# plt.show()


# %%
###############################################################################
######################### SECOND PYTHON SCRIPT ################################
###############################################################################
###############################################################################

# Second Python Script used for doing some of the plots.

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:40:05 2024
With code copied/modified from the educational material handed out in the 
DTU course 02450 - Introduction to Machine Learning and Data Mining.
@author: Christian Damén Schultz-Nielsen, s240369
"""

import importlib_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns
from scipy.linalg import svd




##################### DATA LOAD AND VARIABLE DECLARATION #####################
# %% 
#Cschultz - I placed the wine data in the dtuimldmtools datafolder.. If you 
#don't want to do this, locate your folder with importlib_resources.files!
filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine.data")

wine_data = np.loadtxt(filename, delimiter=",")

D_wineyard = wine_data[:, 0];
D_alcohol = wine_data[:, 1];
D_malic_acid = wine_data[:, 2];
D_ash = wine_data[:, 3];
D_alkalinity_of_ash = wine_data[:, 4];
D_magnesium = wine_data[:, 5];
D_total_phenols = wine_data[:, 6];
D_flavanoids = wine_data[:, 7];
D_non_flavanoid_phenols = wine_data[:, 8];
D_proanthocyanins = wine_data[:, 9];
D_color_int = wine_data[:, 10];
D_hue = wine_data[:, 11];
D_od280_315 = wine_data[:, 12];
D_proline = wine_data[:, 13];

Stat_Means = np.zeros((13,1))
Stat_StdDev = np.zeros((13,1))
Stat_Median = np.zeros((13,1))
Stat_Range = np.zeros((13,1))

for i in range(0, 13-1):
    Stat_Means[i] = np.mean(wine_data[:,i+1])
    Stat_StdDev[i] = np.std(wine_data[:,i+1])
    Stat_Median[i] = np.median(wine_data[:,i+1])
    Stat_Range[i] = np.max(wine_data[:,i+1]) - np.min(wine_data[:,i+1])
''
Stat_Covariance = np.cov(wine_data[2],wine_data[3])
Stat_CorrCoef = np.corrcoef(wine_data[2],wine_data[3])   
    

##################### BASIC PLOTS #####################
# %% 
# Data attributes to be plotted
#Remember i = 0 is the wineyard!! use i>0
plt.close(fig=1)
plt.close(fig=2)
i = 3
j = 12
k = 6
l = 7

Attr_name = ['Cul','Alcohol','Malic Acid','Ash','Alkalinity ash',
             'magnesium','total phenols','flavanoids','non flav. phen.','proanthocy.','color intensity','hue','od280/315','proline']
#
f = plt.figure()

# Create a subplot with 2 plots in a row
plt.subplot(1, 2, 1)  # (rows, columns, index)
plt.plot(wine_data[:, i], wine_data[:, j], "o")
plt.title('Least correlated', fontsize=20)
plt.xlabel(Attr_name[i], fontsize=15)
plt.ylabel(Attr_name[j], fontsize=15)

plt.subplot(1, 2, 2)  # (rows, columns, index)
plt.plot(wine_data[:, k], wine_data[:, l], "o")
plt.title('Most correlated', fontsize=20)
plt.xlabel(Attr_name[k], fontsize=15)
plt.ylabel(Attr_name[l], fontsize=15)

# Adjust layout
plt.tight_layout()


# Make another more fancy plot that includes legend, class labels,
# attribute names, and a title.
f = plt.figure()
plt.title("Wine data")

cultivar = np.array(['alpha','beta','gamma'])
y = wine_data[:,0]


for c in range(1, len(cultivar) + 1):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(wine_data[class_mask, i], wine_data[class_mask, j], "o", alpha=0.3)

plt.legend(cultivar)
plt.xlabel(Attr_name[i])
plt.ylabel(Attr_name[j])

# Output result to screen
plt.show()

##################### HEATMAP #####################
# %%

df = pd.DataFrame({Attr_name[1]: wine_data[:, 1], Attr_name[2]: wine_data[:, 2],
              Attr_name[3]: wine_data[:, 3], Attr_name[4]: wine_data[:, 4], Attr_name[5]: wine_data[:, 5],
              Attr_name[6]: wine_data[:, 6], Attr_name[7]: wine_data[:, 7], Attr_name[8]: wine_data[:, 8],
              Attr_name[9]: wine_data[:, 9], Attr_name[10]: wine_data[:, 10], Attr_name[11]: wine_data[:, 11],
              Attr_name[12]: wine_data[:, 12], Attr_name[13]: wine_data[:, 13]})

correlation_matrix = df.corr()
plt.figure()
sns.heatmap(correlation_matrix, annot=True, annot_kws={'size': 16} ,cmap="coolwarm", fmt=".2f")


# Add labels and title

plt.xlabel('Attributes', fontsize=20)
plt.ylabel('Attributes', fontsize=20)
plt.title('Correlation Heatmap', fontsize=40)
plt.xticks(fontsize=20, rotation = 30)
plt.yticks(fontsize=20, rotation = 0)
# Show the plot
plt.show()



##################### BIG SCATTER PLOT #####################
# %%
M = len(Attr_name)

plt.figure()
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1 * M + m2 + 1)
        for c in range(1,len(cultivar)+1):
            class_mask = y == c
            plt.plot(np.array(wine_data[class_mask, m2]), np.array(wine_data[class_mask, m1]), ".")
            if m1 == M - 1:
                plt.xlabel(Attr_name[m2],fontsize=16, rotation = 45)
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(Attr_name[m1],fontsize=16, rotation = 45)
            else:
                plt.yticks([])
            # ylim(0,X.max()*1.1)
            # xlim(0,X.max()*1.1)
plt.legend(cultivar)
plt.suptitle('Scatterplots between all attributes', fontsize=40)

plt.show()

##################### EXPERIENCED VARIANCE #####################
# %%
# Subtract mean value from data and divide by std. dev.
wine_data_2 = np.delete(wine_data, 0, axis=1)
Attr_name_2 = Attr_name[1:14]

N = len(wine_data_2[:,1])
Y = wine_data_2 - np.ones((N, 1)) * wine_data_2.mean(axis=0)
Y = Y * (1 / np.std(Y, 0))

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components",fontsize=20)
plt.xlabel("Principal component",fontsize=16)
plt.ylabel("Variance explained",fontsize=16)
plt.legend(["Individual", "Cumulative", "Threshold"],fontsize=16)
plt.grid()
plt.show()

##################### PCA COMPONENT DIRECTIONS #####################
# %%

V = V.T
N, M = wine_data_2.shape
plt.figure()
# We saw that the first 8 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0, 1, 2, 3]
legendStrs = ["PC" + str(e + 1) for e in pcs]
#c = ["r", "g", "b"]
bw = 0.15
r = np.arange(1, M + 1)
for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)
plt.xticks(r + bw, Attr_name_2)
plt.xlabel("Attributes",fontsize=16)
plt.ylabel("Component coefficients",fontsize=20)
plt.legend(legendStrs,fontsize=16)
plt.grid()
plt.xticks(fontsize=16, rotation=30)
plt.yticks(fontsize=16)  
plt.title("PCA Component Coefficients of Wine Data",fontsize=25)
plt.show()

###############################################################################
###############################################################################