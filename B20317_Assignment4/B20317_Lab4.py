"""
NAME       - SARTHAK SAPTAMI KUMAR JHA
ROLL NO.   - B20317
MOBILE NO. - 8825319259
BRANCH     - Computer Science and Engineering (CSE)
"""
import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn
import math
import statistics as stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from decimal import Decimal


def manual_KNN(k_val, df_test, df_train):
    S_final = []
    for i in range(len(df_test)):
        K_1 = []
        Final = []
        A = df_test.iloc[i, [t for t in range(0, 27)]].tolist()
        for j in range(len(df_train)):
            s = 0
            B = df_train.iloc[j, [t for t in range(0, 27)]].tolist()
            for m in range(0, 27):
                s += (A[m] - B[m]) ** 2
            a = s ** 0.5
            K_1.append(a)
        K_1 = np.array(K_1)
        location = (K_1.argsort()[:k_val])
        for cl in range(k_val):
            Final.append(df_train['Class'].iloc[location[cl]])
        c_0 = Final.count(0)
        c_1 = Final.count(1)
        if c_0 > c_1:
            S_final.append(0)
        else:
            S_final.append(1)
    return S_final



# Create a dataframe

df = pd.read_csv("SteelPlateFaults-2class.csv")

# Create lists of attributes
Attributes_with_class = df.columns.values.tolist()
Attributes = Attributes_with_class[:len(Attributes_with_class)-1]


# Create x and y containg class and other attributes
y = df['Class'].tolist()
x = []
for i in range(len(df)):
    x.append(df.iloc[i, [i for i in range(0, 28)]].tolist())


# Using x and y in splitting data into test and training sets
[x_train, x_test, x_label_train, x_label_test] = train_test_split(x, y, test_size=0.3, random_state=42,shuffle=True)

# Creating dataframes for test and train data
df_test = pd.DataFrame(x_test, columns = Attributes_with_class)
df_train = pd.DataFrame(x_train, columns = Attributes_with_class)


# List of class variables in original data for comparision
orig = df_test['Class'].tolist()

switch = 0

# Manual KNN
if switch == 1:
    for i in range(1, 6, 2):
        S_final = manual_KNN(3, df_test, df_train)
        print(confusion_matrix(orig, S_final))
        print(accuracy_score(orig, S_final))

# In-Built KNN
for k_val in range(1, 6, 2):

    knn = KNeighborsClassifier(n_neighbors=k_val)
    knn.fit(x_train, x_label_train)
    A = knn.predict(x_test).tolist()
    print('Covariance matrix and accuracy score for K =', k_val)
    print(confusion_matrix(orig, A))
    print(accuracy_score(orig, A))
    print()

#---------------------------------------------------------------------------------------------------------------------------------------------------
# Normalizing data before KNN

# Storing the max and min values for later use with test data
train_min = df_train.min().tolist()
train_max = df_train.max().tolist()

# Normalizing training data
df_train_n = df_train.copy()
for column in Attributes:
    df_train_n[column] = ((df_train_n[column] - df_train_n[column].min()) / (df_train_n[column].max() - df_train_n[column].min()))

# Normalizing test data
it = 0
df_test_n = df_test.copy()
for column in Attributes:
    df_test_n[column] = ((df_test_n[column] - train_min[it]) / (train_max[it] - train_min[it]))
    it += 1

# Converting normalized test and train dataframes into lists for input into KNN
x_train = []
x_test = []
for i in range(len(df_train_n)):
    x_train.append(df_train_n.iloc[i, [s for s in range(0, 27)]].tolist())
x_label_train = df_train_n['Class'].tolist()

for i in range(len(df_test_n)):
    x_test.append(df_test_n.iloc[i, [s for s in range(0, 27)]].tolist())

# Calculating KNN for the normalized data
for k_val in range(1, 6, 2):
    knn = KNeighborsClassifier(n_neighbors=k_val)
    knn.fit(x_train, x_label_train)
    A = knn.predict(x_test).tolist()

    print('Covariance matrix and accuracy score for K =', k_val)
    orig = df_test_n['Class'].tolist()
    print(confusion_matrix(orig, A))
    print(accuracy_score(orig, A))
    print()

# ------------------------------------------------------------------------------------------------------------------------------

# Dividing train data by class
df_train_0 = df_train[df_train['Class'] == 0]
df_train_1 = df_train[df_train['Class'] == 1]

# Making shallow copies
df_train_0_nc = df_train_0[:]
df_train_1_nc = df_train_1[:]
df_test_nc = df_test[:]

# Removing class for calculation of mean and covariance vectors
df_train_0_onc = df_train_0.drop(['Class'], axis = 1)
df_train_1_onc = df_train_1.drop(['Class'], axis = 1)

# Mean vectors
print('Mean of Class 0:')
print(df_train_0_onc.mean())
print()
print('Mean of Class 1:')
print(df_train_1_onc.mean())
print()

# Covariance matrix
print('Covariance matrix of Class 0:')
print(np.cov(df_train_0_nc.T))
print()
print('Covariance matrix of Class 1:')
print(np.cov(df_train_1_onc.T))
print()

anc1 = np.cov(df.drop(['Class'], axis = 1).T)
cov_df_1 = pd.DataFrame(anc1, columns = [i for i in range(1, 28)], index = [i for i in range(1, 28)])


# Removing various attributes to avoid singularity for calculation of mean and covariance vectors
df_train_0_nc = df_train_0_nc.drop(['Class','X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis = 1)
df_train_1_nc = df_train_1_nc.drop(['Class','X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis = 1)
df_test_nc = df_test_nc.drop(['Class', 'X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis = 1)

# Mean vectors
df_train_0_mean = df_train_0_nc.mean()
df_train_1_mean = df_train_1_nc.mean()

# Covariance matrix
cov_train_0 = np.cov(df_train_0_nc.T)
cov_train_1 = np.cov(df_train_1_nc.T)
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})


Final = []

# Prior probability of each class
pre_1 = len(df_test[df_test['Class'] == 1]) / len(df_test)
pre_0 = len(df_test[df_test['Class'] == 0]) / len(df_test)



for i in range(len(df_test_nc)):
    # Creating a data vector for each row
    A = df_test_nc.iloc[i, [t for t in range(0, 23)]].tolist()
    A = np.array(A)
    A = A.reshape(23, 1)

    # Calculating and creating mean vectors for each class
    df_train_0_mean = np.array(df_train_0_mean)
    df_train_0_mean = df_train_0_mean.reshape(23, 1)

    df_train_1_mean = np.array(df_train_1_mean)
    df_train_1_mean = df_train_1_mean.reshape(23, 1)

    # Computing the likelihood of the test vector for each class
    f1 = 1.0/(math.pow((2*np.pi), float(23)/2) * math.pow(np.linalg.det(cov_train_0), (1.0/2)))
    f2 = np.dot(((A - df_train_0_mean).T), (np.linalg.inv(cov_train_0)))
    f3 = np.dot(f2, (A - df_train_0_mean))
    # Likelihood of test data being in class 0 is f4
    f4 = f1 * (math.pow(math.e, (-0.5 * float(f3[0][0]))))

    g1 = 1.0/(math.pow((2*np.pi),float(23)/2) * math.pow(np.linalg.det(cov_train_1), (1.0/2)))
    g2 = np.dot(((A - df_train_1_mean).T), (np.linalg.inv(cov_train_1)))
    g3 = np.dot(f2, (A - df_train_1_mean))
    # Likelihood of test data being in class 1 is g4
    g4 = g1 * (math.pow(math.e, (-0.5 * float(g3[0][0]))))


    # Calculating the posterior probabilities and assigning classes

    if f4 != 0 and g4 != 0:
        sum = pre_0 * f4 + pre_1 * g4

        pro_1 = (pre_1 * g4) / sum
        pro_0 = (pre_0 * f4) / sum

        # Checking which class the test data belongs to
        if pro_0 > pro_1:
            Final.append(0)
        else:
            Final.append(1)

    # If the values of likelihood are very low, it is converted to 0.0, to avoid this, we use
    # Decimal function in python to get precision in very small values
    else:
        (f4) = Decimal(2.71828) ** (Decimal(-0.5) * Decimal(f3[0][0]))
        (g4) = Decimal(2.71828) ** (Decimal(-0.5) * Decimal(g3[0][0]))

        f4 = Decimal(f4) * Decimal(f1)
        g4 = Decimal(g4) * Decimal(g1)

        (sum) = Decimal(pre_0) * Decimal(f4) + Decimal(pre_1) * Decimal(g4)

        (pro_1) = (Decimal(pre_1) * Decimal(g4)) / Decimal(sum)
        (pro_0) = (Decimal(pre_0) * Decimal(f4)) / Decimal(sum)

        if pro_0 > pro_1:
            Final.append(0)
        else:
            Final.append(1)

# Class variables of original data, to be used for comparision
orig = df_test['Class'].tolist()

# Printing confusion matrix and accuracy of the bayes classifier
print('Covariance matrix and accuracy score for Bayes Classifier')
print(confusion_matrix(orig, Final))
print(accuracy_score(orig, Final))
print()


