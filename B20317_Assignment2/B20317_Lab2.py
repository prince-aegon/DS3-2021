# -*- coding: utf-8 -*-
"""
NAME       - SARTHAK SAPTAMI KUMAR JHA
ROLL NO.   - B20317
MOBILE NO. - 8825319259
BRANCH     - CSE
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate RMSE for filling by mean
def rmse(st, l):
    ta = df_m[st].mean()
    s = 0
    count = 1
    for i in range(len(l)):
        k = dg[st]
        p = df_m[st]
        ta = p.loc[l[i]]
        v1 = k.loc[l[i]]
        s+= (ta - v1) ** 2
        count+=1
    return ((s/len(l)) ** 0.5)

# Function to calculate RMSE for filling by interpolation
def rmse_i(st, l):
    ta = df_i[st].mean()
    s = 0
    count = 1
    for i in range(len(l)):
        k = dg[st]
        p = df_i[st]
        ta = p.loc[l[i]]
        v1 = k.loc[l[i]]
        s+= (ta - v1) ** 2
        count+=1
    return ((s/len(l)) ** 0.5)

# Function to calculate statistics for original file
def stats_original():
    print('Original Mean values: ')
    print()
    print(dg.mean())
    print('Original Median values: ')
    print()
    print(dg.median())
    print('Original Standard Deviation values: ')
    print()
    print(dg.std())
    print('Original Mode values: ')
    print()
    B = ['dates', 'stationid', 'temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
    for i in B:
        print(i, dg[i].mode())

# Function to calculate statistics after filling missing values by mean
def stats_mean():
    print('New Mean values: ')
    print(df_m.mean())
    print()
    print('New Median values: ')
    print(df_m.median())
    print()
    print('New Standard Deviation values: ')
    print()
    print(df_m.std())
    print()
    print('New Mode values: ')
    print()
    B = ['dates', 'stationid', 'temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
    for i in B:
        print(i, df_m[i].mode())
    print()

# Function to calculate statistics after filling missing values by interpolation

def stats_interpolation():
    print('New Mean values: ')
    print(df_i.mean())
    print()
    print('New Median values: ')
    print(df_i.median())
    print()
    print('New Standard Deviation values: ')
    print()
    print(df_i.std())
    print()
    print('New Mode values: ')
    print()
    B = ['dates', 'stationid', 'temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
    for i in B:
        print(i, df_i[i].mode())
    print()
    
# Function to print boxplot for before removing outliers
def outlier_boxplot_1(st):
    plt.boxplot(df_i[st])
    plt.show()

# Function to detect and replace outliers by their respective median values
def outlier_detect(st):
    
    q3 = np.quantile(df_i[st], 0.75)
    q1 = np.quantile(df_i[st], 0.25)
    iqr = q3 - q1
    lr = q1 - (1.5*iqr)
    ur = q3 + (1.5*iqr)
    
    
    ol = [i for i in df_i[st] if ((i < lr) | (i > ur))]
    # print(ol)
    
    m = df_i[st].median()
    A = []
    for i in range(len(df_i[st])):
        x = (df_i[st].iloc[i])
        if ((x < lr) or (x > ur)):
            A.append(x)
            df_i[st] = df_i[st].replace(x, m)
            
            
    q3 = np.quantile(df_i[st], 0.75)
    q1 = np.quantile(df_i[st], 0.25)
    iqr = q3 - q1
    lr = q1 - (1.5*iqr)
    ur = q3 + (1.5*iqr)        
            
    
    oli = [i for i in df_i[st] if ((i < lr) | (i > ur))]
    # print(oli)
    plt.boxplot(df_i[st])
    plt.show()




# Creating dataframes for the csv files
df = pd.read_csv("landslide_data3_miss.csv")
dg = pd.read_csv("landslide_data3_original.csv")





# Graph of attribute names and the number of missing values in each.
A = df.isnull().sum()
A = A.to_list()

B = ['dates', 'stationid', 'temperature', 'humidity', 'pressure', 'rain', 'lightavgw', 'lightmax', 'moisture']

plt.bar(B, A, width = 0.8)
plt.xticks(rotation = 30)
plt.xlabel('Attributes')
plt.ylabel('Number of missing values')
plt.title('Number of missing values vs attributes')
plt.show()





original_n = len(df)

# Dropping tuples with missing 'stationid'
df = df.dropna(how='any', subset=['stationid'])

new_n1 = len(df)

print('Total number of tuples deleted in first time:', original_n - new_n1)
print()


# Dropping tuples after removing tuples with 1/3rd missing values.
df = df.dropna(thresh=6)

new_n2 = len(df)

print('Total number of tuples deleted in second time:', new_n1 - new_n2)
print()


# Printing the total number of missing values in each attribute after deletion
A = df.isnull().sum()
A = A.to_list()

count = 0
B = ['dates', 'stationid', 'temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
for i in B:
    s = 'Number of missing values in ' + i +' is: ' + str(A[count])
    print(s)
    count += 1
print()
print('Total number of missing values:', sum(A))
print()



# Creating shallow copies each for filling with mean and interpolation.
df_m = df.copy(deep=False)
df_i = df.copy(deep=False)




# Filling missing values with mean.

stats_original()

# Creating a list of indices of all the missing values, for later calculation of RMSE
supl = []

B = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
for i in B:
    supl.append(df_m[df_m[i].isnull()].index.tolist())


# List to store RMSE values
supo = []

# Filling missing values with mean
df_m = df_m.fillna(df_m.mean())


# Printing stats after filling data with mean.
stats_mean()

# Calculating RMSE for all attributes
count = 0
B = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
for i in B:
    supo.append(rmse(i, supl[count]))
    count = count + 1
    
# Printing RMSE Values
B = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
for i in range(len(B)):
    print(B[i], '=', supo[i])


# Plotting RMSE values
plt.bar(B, supo)
plt.xticks(rotation = 30)
plt.xlabel('Attributes')
plt.ylabel('RMSE')
plt.title('RMSE vs attributes')
plt.show()



# Filling missing values using interpolation



# Creating a list of indices of all the missing values, for later calculation of RMSE
supl = []
B = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
for i in B:
    supl.append(df_i[df_i[i].isnull()].index.tolist())


# List to store RMSE values
supo = []


# Filling missing values using interpolation
df_i = df_i.interpolate(method ='linear')


# Printing stats after interpolation
stats_interpolation()


# Calculating RMSE for all attributes
count = 0
B = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
for i in B:
    supo.append(rmse_i(i, supl[count]))
    count = count + 1
    
# Printing RMSE values
B = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']
for i in range(len(B)):
    print(B[i], '=', supo[i])


# Plotting RMSE values
plt.bar(B, supo)
plt.xticks(rotation = 30)
plt.xlabel('Attributes')
plt.ylabel('RMSE')
plt.title('RMSE vs attributes')
plt.show()




# Outlier detection
B = ['temperature', 'rain']
for st in B:
    outlier_boxplot_1(st)
    outlier_detect(st)
