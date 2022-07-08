# -*- coding: utf-8 -*-
"""
NAME       - SARTHAK SAPTAMI KUMAR JHA
ROLL NO.   - B20317
MOBILE NO. - 8825319259
"""

import pandas as pd
import matplotlib.pyplot as plt

# Function for Mean

def E(A):
    s = 0
    for i in A:
        s += i
    return s/len(A)

# Function for variance

def Var(A):
    e = E(A)
    s = 0
    for i in A:
        s += (i - e) ** 2
    return s/(len(A)-1)

# Function for standard deviation
def SD(A):
    return (Var(A) ** 0.5)

# Function for covariance
def Cov(A, B):
    s = 0
    ea = E(A)
    eb = E(B)
    for i in range(len(A)):
        s += (A[i] - ea) * (B[i] - eb)
    return s/(len(A)-1)

# Function for correlation
def Corr(A, B):
    return Cov(A, B)/(SD(A) * SD(B))





# Function to plot scatterplots
def scatterplot(s1, s2):
    plt.xlabel(s1)
    plt.ylabel(s2) 
    title = 'Scatter plot: '+s1+' vs. '+s2
    plt.title(title)
    plt.scatter(df[s1], df[s2], s=20)
    plt.show()
    
# Function to calculate correlation coefficient
def correlation(s1, s2):
    A = df[s1].tolist()
    B = df[s2].tolist()
    k = round(Corr(A, B), 3)
    s = 'Correlation between '+s1+' and '+s2+' is: '+str(k)
    print(s)
    
# Function to plot boxplot
def boxplot(s):
    s1 = s + ' value'
    plt.ylabel(s1) 
    s2 = 'Boxplot of ' + s
    plt.title(s2)
    plt.boxplot(df[s])
    plt.show()




# Importing the csv file into a pandas dataframe

df = pd.read_csv("pima-indians-diabetes.csv")


# 1. Using various pandas functions to caluclate the mean, median, mode, etc. of the dataframe.

mean_df = df.mean()
median_df = df.median()
mode_df = df.mode()
min_df = df.min()
max_df = df.max()
stdev_df = df.std()


# Printing mean, median, mode, etc. of all the attributes

values = [mean_df, median_df, mode_df, min_df, max_df, stdev_df]
names = ['mean', 'median', 'mode', 'minimum', 'maximum', 'standard deviation']

for i in range(len(values)):
    s = 'The ' + names[i] + ' of all attributes are:'
    print(s)
    print(values[i])
    
print('-----------------------------------------------------')
    

# 2. a. Scatter plot between 'Age' and other attributes

Attributes = ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi']
s1 = 'Age'

for i in Attributes:
    scatterplot(s1,i)




# 2. b. Scatter plot between 'BMI' and other attributes

Attributes = ['pregs', 'plas', 'pres', 'skin', 'test', 'pedi', 'Age']
s2 = 'BMI'

for i in Attributes:
    scatterplot(s2,i)




# 3. a. Correlation between 'Age' and other attributes

Attributes = ['pregs', 'plas', 'pres', 'skin', 'test','BMI', 'pedi', 'Age']
s2 = 'Age'

for i in Attributes:
    correlation(i, s2)


print('------------------------------------------------------')


# 3. b. Correlation between 'BMI' and other attributes

Attributes = ['pregs', 'plas', 'pres', 'skin', 'test','BMI', 'pedi', 'Age']
s2 = 'BMI'

for i in Attributes:
    correlation(i, s2)




# 4. Histogram depiction of attribute pregs and skin

plt.xlabel('pregs')
plt.ylabel('Frequency') 
plt.title('Histogram plot: pregs')
plt.hist(df['pregs'], bins = 15)
plt.show()

plt.xlabel('skin')
plt.ylabel('Frequency') 
plt.title('Histogram plot: skin')
plt.hist(df['skin'], bins = 15)
plt.show()




# 5. Histogram depiction of attribute pregs relatove to classes

da = df[df['class'] == 0]
plt.xlabel('pregs')
plt.ylabel('Frequency') 
plt.title('Histogram plot: pregs & class = 0')
plt.hist(da['pregs'], bins = 12)
plt.show()

db = df[df['class'] == 1]
plt.xlabel('pregs')
plt.ylabel('Frequency') 
plt.title('Histogram plot: pregs & class = 1')
plt.hist(db['pregs'], bins = 15)
plt.show()



# 6. Boxplot of all attributes

Attributes = ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi', 'Age']

for i in Attributes:
    boxplot(i)
    
    
# End of file