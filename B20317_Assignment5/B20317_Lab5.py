"""
NAME       - SARTHAK SAPTAMI KUMAR JHA
ROLL NO.   - B20317
MOBILE NO. - 8825319259
BRANCH     - Computer Science and Engineering (CSE)
"""

import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import math
import statistics as stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from decimal import Decimal
import sklearn.mixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



def uni_lin_best_fit_line(x_train_shell_np, x_label_train):
    plt.figure(1)
    m, b = np.polyfit(x_train_shell_np, x_label_train, 1)
    plt.plot(x_train_shell_np, x_label_train, 'o')
    plt.plot(x_train_shell_np, m * x_train_shell_np + b)
    plt.xlabel('Shell-Weight')
    plt.ylabel('Rings')
    plt.title('Best fit line for simple linear regression model')
    plt.show()


def scatter_1(x_label_test, y_pred):
    plt.figure(2)
    plt.scatter(x_label_test, y_pred)
    plt.xlabel('Actual Rings')
    plt.ylabel('Predicted Rings')
    plt.title('Actual vs Predicted Rings')
    plt.show()

def scatter_2(x_label_test, pred_y):
    plt.figure(3)
    plt.scatter(x_label_test, pred_y)
    plt.xlabel('Actual Rings')
    plt.ylabel('Predicted Rings')
    plt.title('Actual vs Predicted Rings')
    plt.show()

def RMSE_1(rmse):
    plt.figure(4)
    plt.xlabel('Degree of the polynomial')
    plt.ylabel('RMSE Values')
    plt.title('RMSE vs Degree of the polynomial')
    plt.plot([i for i in range(2, 6)], rmse)
    plt.show()

def RMSE_2(rmse):
    plt.figure(5)
    plt.xlabel('Degree of the polynomial')
    plt.ylabel('RMSE Values')
    plt.title('RMSE vs Degree of the polynomial')
    plt.plot([i for i in range(2, 6)], rmse)
    plt.show()

def non_lin_best_fit_line(newx, newy, x_train_shell, x_label_train):
    plt.figure(6)
    plt.plot(newx, newy, color = 'm')
    plt.scatter(x_train_shell, x_label_train)
    plt.xlabel('Shell weight')
    plt.ylabel('Rings')
    plt.title("Polynomial Degree = 4")
    plt.show()

def scatter_3(x_label_test, pred_y):
    plt.figure(7)
    plt.scatter(x_label_test, pred_y)
    plt.xlabel('Actual Number of Rings')
    plt.ylabel('Predicted number of Rings')
    plt.title("Actual vs Predicted number of Rings")
    plt.show()

def RMSE_3(rmse):
    plt.figure(8)
    plt.xlabel('Degree of the polynomial')
    plt.ylabel('RMSE Values')
    plt.title('RMSE vs Degree of the polynomial')
    plt.plot([i for i in range(2, 6)], rmse)
    plt.show()

def RMSE_4(rmse):
    plt.figure(9)
    plt.xlabel('Degree of the polynomial')
    plt.ylabel('RMSE Values')
    plt.title('RMSE vs Degree of the polynomial')
    plt.plot([i for i in range(2, 6)], rmse)
    plt.show()

def scatter_4(x_label_test, pred_y):
    plt.figure(10)
    plt.scatter(x_label_test, pred_y)
    plt.xlabel('Actual Number of Rings')
    plt.ylabel('Predicted number of Rings')
    plt.title("Actual vs Predicted number of Rings")
    plt.show()


# PART A
# Create a dataframe
df = pd.read_csv("SteelPlateFaults-2class.csv")

# Dropping attributes with high correlation
df = df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis = 1)

# Create lists of attributes
Attributes_with_class = df.columns.values.tolist()
Attributes = Attributes_with_class[:len(Attributes_with_class)-1]


# Create x and y containg class and other attributes
y = df['Class'].tolist()
x = []
for i in range(len(df)):
    x.append(df.iloc[i, [i for i in range(0, 24)]].tolist())


# Using x and y in splitting data into test and training sets
[x_train, x_test, x_label_train, x_label_test] = train_test_split(x, y, test_size=0.3, random_state=42)

# Creating dataframes for test and train data
df_test = pd.DataFrame(x_test, columns = Attributes_with_class)
df_train = pd.DataFrame(x_train, columns = Attributes_with_class)


# List of class variables in original test data for comparison
orig = df_test['Class'].tolist()


# Creating list of class 0 samples and class 1 samples in train data
df_train_0 = df_train[df_train['Class'] == 0]
df_train_1 = df_train[df_train['Class'] == 1]

x_train_0 = df_train_0.values.tolist()
x_train_1 = df_train_1.values.tolist()

# Conducting Bayes with GMM for differnet values of Q
for Q in [2, 4, 8, 16]:
    # Training 2 independent sklearn GMMs on the data points associated with each label
    GMM0 = sklearn.mixture.GaussianMixture(n_components=Q, covariance_type='full')
    GMM0.fit(x_train_0)

    GMM1 = sklearn.mixture.GaussianMixture(n_components=Q, covariance_type='full')
    GMM1.fit(x_train_1)

    # Calculating weighted log probabilities for each sample
    s0 = GMM0.score_samples(x_test)
    s1 = GMM1.score_samples(x_test)

    f = []
    for i in range(len(df_test)):
        # For each new sample, we make use the score_samples data to check which class the
        # new data belongs to.
        if s0[i] > s1[i]:
            f.append(0)
        else:
            f.append(1)

    # Printing the confusion matrix and accuracy score
    print('For Q =', Q)
    print('Confusion matrix: ')
    print(confusion_matrix(orig, f))
    print('Classification Accuracy', round(accuracy_score(orig, f), 4))
    print()

# ------------------------------------------------------------------------------------------------------
# PART B

# Creating a dataframe
df = pd.read_csv("abalone.csv")

# Creating a list of columns
Att = df.columns.tolist()

# Creating list of data and Rings
x = []
y = []
for i in range(len(df)):
    x.append(df.iloc[i, [i for i in range(0, 7)]].tolist())
y = df['Rings'].tolist()

# Calculating the attribute with highest correlation to Rings
corr = []
for i in Att[:len(Att)-1]:
    corr.append(df[i].corr(df['Rings']))
for i in Att:
    if max(corr) == df[i].corr(df['Rings']):
        print('Max correlation is between Rings and', i)
print()
# We find that shell-weight has highest correlation with Rings

# Using x and y in splitting data into test and training sets
[x_train, x_test, x_label_train, x_label_test] = train_test_split(x, y, test_size=0.3, random_state=42,shuffle=True)

# Creating np arrays and lists of Shell-weight in train and test data
df_test = pd.DataFrame(x_test, columns = Att[:7])
df_train = pd.DataFrame(x_train, columns = Att[:7])

x_train_shell = df_train['Shell weight'].tolist()
x_test_shell  = df_test['Shell weight'].tolist()

x_train_shell_np = np.array(x_train_shell)
x_test_shell_np = np.array(x_test_shell)

# Drawing the best fit line
uni_lin_best_fit_line(x_train_shell_np, x_label_train)


# Reshaping the arrays shell-weight train and shell-weight test
x_train_shell = [[i] for i in x_train_shell]
x_test_shell = [[i] for i in x_test_shell]

# Question 1
# Conducting Linear Regression
reg = LinearRegression()

# Fitting Training data and Training labels
reg.fit(x_train_shell, x_label_train)

# Predicting values for Test data
y_pred = reg.predict(x_test_shell)
# Predicting values for Training data
y_pred_train = reg.predict(x_train_shell)

print('RMSE for Question 1')
# Printing RMSE error values for prediction data and test labels
mse = sklearn.metrics.mean_squared_error(x_label_test, y_pred, squared = False)
print("Root Mean Squared Error for Test data {}".format(mse))

# Printing RMSE error values for prediction data and train labels
mse = sklearn.metrics.mean_squared_error(x_label_train, y_pred_train, squared = False)
print("Root Mean Squared Error for Train data {}".format(mse))
print()

# Plotting scatter plot between predicted and actual rings
scatter_1(x_label_test, y_pred)



# Question 2
# Conducting Linear Regression
lr = LinearRegression()

# Fitting complete train data and training labels
lr.fit(x_train, x_label_train)

# Predicting values for test data
pred_y = lr.predict(x_test)


print('RMSE for Question 2')
# Printing RMSE error values for prediction data and test labels
mse = sklearn.metrics.mean_squared_error(x_label_test, pred_y, squared = False)
print("Root Mean Squared Error for Test data {}".format(mse))

# Plotting scatter plot between predicted and actual rings for test data
scatter_2(x_label_test, pred_y)

# Predicting values for Training data
pred_y_train = lr.predict(x_train)

# Printing RMSE error values for prediction data and training labels
mse = sklearn.metrics.mean_squared_error(x_label_train, pred_y_train, squared = False)
print("Root Mean Squared Error for Train data {}".format(mse))
print()


# Question 3
print('Question 3')

# Conducting Non-Linear Regression for various values of p


# Conducting regression and predicting TRAINING values
# List to store RMSE values
rmse = []

for degree in range(2, 6):

    # Conducting Regression for p -> degree
    poly_features = PolynomialFeatures(degree = degree)

    # Fitting data
    x_poly = poly_features.fit_transform(x_train_shell)
    train = poly_features.fit_transform(x_train_shell)

    # Conducting Regression
    regressor = LinearRegression()
    regressor.fit(x_poly, x_label_train)

    # Predicting Training values
    pred_y = regressor.predict(train)

    # Calculating RMSE values
    mse = sklearn.metrics.mean_squared_error(x_label_train, pred_y, squared = False)

    # Storing RMSE values in the list
    rmse.append(mse)

# Plotting RMSE vs degree of polynomial for TRAIN data
RMSE_1(rmse)


# Conducting regression and predicting TESTING values
# List to store RMSE values
rmse = []
for degree in range(2, 6):


    # Conducting Regression for p -> degree
    poly_features = PolynomialFeatures(degree = degree)

    # Fitting data
    x_poly = poly_features.fit_transform(x_train_shell)
    test = poly_features.fit_transform(x_test_shell)

    # Conducting Regression
    regressor = LinearRegression()
    regressor.fit(x_poly, x_label_train)

    # Predicting Testing values
    pred_y = regressor.predict(test)

    # Calculating RMSE values
    mse = sklearn.metrics.mean_squared_error(x_label_test, pred_y, squared = False)

    # Storing RMSE values in the list
    rmse.append(mse)

# Plotting RMSE vs degree of polynomial for TEST data
RMSE_2(rmse)


# Calculating for which value of p RMSE error is minimum
for i in range(len(rmse)):
    if min(rmse) == rmse[i]:
        print('Minimum RMSE is for p =', i+2)


# Repeating Regression process on TRAIN data with degree 4 for best fit line
# Degree 4 is chosen as it has lowest RMSE for TEST data


poly_features = PolynomialFeatures(degree = 4)
x_poly = poly_features.fit_transform(x_train_shell)
train = poly_features.fit_transform(x_train_shell)

regressor = LinearRegression()
regressor.fit(x_poly, x_label_train)

# Predicting values for train data
pred_y = regressor.predict(train)

# Sorting data for plotting best fit line
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x_train_shell, pred_y), key=sort_axis)
newx, newy = zip(*sorted_zip)

# Plotting best fit line with scatter plot
non_lin_best_fit_line(newx, newy, x_train_shell, x_label_train)


# Repeating Regression process on TEST data with degree 4 for scatter plot
# Degree 4 is chosen as it has lowest RMSE for TEST data

poly_features = PolynomialFeatures(degree = 4)
x_poly = poly_features.fit_transform(x_train_shell)
test = poly_features.fit_transform(x_test_shell)

regressor = LinearRegression()
regressor.fit(x_poly, x_label_train)

# Predicting values for TEST data
pred_y = regressor.predict(test)

# Plotting scatter plot for predicted and actual data
scatter_3(x_label_test, pred_y)



# Question 4
print()
print('Question 4')

# Coudcting Non-Linear Regression on complete data for various values of p

# Conducting regression and predicting TRAINING values
# List to store RMSE values
rmse = []
for degree in range(2, 6):

    # Conducting Regression for p -> degree
    poly_features = PolynomialFeatures(degree = degree)

    # Fitting data
    x_poly = poly_features.fit_transform(x_train)
    train = poly_features.fit_transform(x_train)

    # Conudcting Regression
    regressor = LinearRegression()
    regressor.fit(x_poly, x_label_train)

    # Predicting Training values
    pred_y = regressor.predict(train)

    # Calculating RMSE values
    mse = sklearn.metrics.mean_squared_error(x_label_train, pred_y, squared  = False)

    # Storing RMSE values in a list
    rmse.append(mse)

# Plotting RMSE values for TRAIN data
RMSE_3(rmse)


# Conducting regression and predicting TESTING values
# List to store RMSE values
rmse = []
for degree in range(2, 6):

    # Conducting Regression for p -> degree
    poly_features = PolynomialFeatures(degree = degree)

    # Fitting data
    x_poly = poly_features.fit_transform(x_train)
    test = poly_features.fit_transform(x_test)

    # Conudcting Regression
    regressor = LinearRegression()
    regressor.fit(x_poly, x_label_train)

    # Predicting Testing values
    pred_y = regressor.predict(test)

    # Calculating RMSE values
    mse = sklearn.metrics.mean_squared_error(x_label_test, pred_y, squared  = False)

    # Storing RMSE values in a list
    rmse.append(mse)

# Plotting RMSE values for Test data
RMSE_4(rmse)

# Calculating for which value of p RMSE error is minimum
for i in range(len(rmse)):
    if min(rmse) == rmse[i]:
        print('Minimum RMSE is for p =', i+2)


# Repeating Regression process on TEST data with degree 2 for scatter plot
# Degree 2 is chosen as it has lowest RMSE for TEST data

poly_features = PolynomialFeatures(degree = 2)

x_poly = poly_features.fit_transform(x_train)
test = poly_features.fit_transform(x_test)

regressor = LinearRegression()
regressor.fit(x_poly, x_label_train)

# Predicting values for TEST data
pred_y = regressor.predict(test)

# Plotting scatter plot for predicted and actual data
scatter_4(x_label_test, pred_y)
