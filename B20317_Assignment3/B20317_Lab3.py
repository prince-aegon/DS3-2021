"""
NAME       - SARTHAK SAPTAMI KUMAR JHA
ROLL NO.   - B20317
MOBILE NO. - 8825319259
BRANCH     - CSE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
from shapely.geometry import Point
from shapely.geometry import LineString
from sklearn.decomposition import PCA
from numpy import linalg as LA
from scipy.linalg import eigh
from sklearn import preprocessing


# Check if there are any outliers
def outlier_check(s):
    A = df[s].tolist()

    q3 = np.quantile(A, 0.75)
    q1 = np.quantile(A, 0.25)
    iqr = q3 - q1
    lr = q1 - (1.5 * iqr)
    ur = q3 + (1.5 * iqr)

    B = []

    for i in A:
        if i > ur or i < lr:
            B.append(i)
    print(s, " ", len(B))

# Calculate median for data without outliers
def median_calc(s):
    A = df[s].tolist()

    q3 = np.quantile(A, 0.75)
    q1 = np.quantile(A, 0.25)
    iqr = q3 - q1
    lr = q1 - (1.5 * iqr)
    ur = q3 + (1.5 * iqr)

    B = []
    for i in A:
        if i > lr and i < ur:
            B.append(i)

    median_replace(s, lr, ur, stats.median(B))

# Replace outliers with median
def median_replace(s, lr, ur, med):
    df[s] = np.where((df[s] > ur), med, df[s])
    df[s] = np.where((df[s] < lr), med, df[s])



def eigen_analysis(df_a):
    A = np.cov(df_a.T)
    w, v = LA.eig(A)
    plt.figure(7)
    plt.xlabel('Eigenvalues')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of the eigenvalues')
    plt.plot([i for i in range(1, 9)], w)
    plt.show()


def reconstruction_error(df, da, l):
    if l == 8:
        return 0
    err = []
    for ind in df.index:
        s = 0
        for i in Attributes:
            s += (df[i][ind] ** 2 - da[i][ind] ** 2)
        err.append(s ** (1 / 2))
    return (sum(err) / len(err))

def plot_2a(x, y):
    # Plotting the scatter plot of the data
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot of Synthetic data')
    plt.plot(x, y, 'x', zorder=1)
    plt.axis('equal')
    plt.show()

def plot_quiver_line(x, y, eig_vec1, eig_vec2):
    plt.figure(2)
    plt.plot(x, y, 'x', zorder=1)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot with eigenvectors')
    plt.quiver(0, 0, eig_vec1[0], eig_vec1[1], color=['r'], scale=2.2, zorder=2)
    plt.quiver(0, 0, eig_vec2[0], eig_vec2[1], color=['r'], scale=2.2 * 3.5, zorder=3)
    plt.show()

def plot_3a(x, y, eig_vec1, eig_vec2, n_x, n_y):
    plt.figure(3)
    plt.plot(x, y, 'x', zorder=1)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot with projection on 1st eigen projection')
    plt.quiver(0, 0, eig_vec1[0], eig_vec1[1], color=['r'], scale=2.2, zorder=2)
    plt.quiver(0, 0, eig_vec2[0], eig_vec2[1], color=['r'], scale=2.2 * 3.5, zorder=3)
    plt.plot(n_x, n_y, 'x', zorder=4)
    plt.show()

def plot_3b(x, y, eig_vec1, eig_vec2, m_x, m_y):
    plt.figure(4)
    plt.plot(x, y, 'x', zorder=1)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot with projection on 2nd eigen projection')
    plt.quiver(0, 0, eig_vec1[0], eig_vec1[1], color=['r'], scale=2.2, zorder=2)
    plt.quiver(0, 0, eig_vec2[0], eig_vec2[1], color=['r'], scale=2.2 * 3.5, zorder=3)
    plt.plot(m_x, m_y, 'x', zorder=5)
    plt.show()

def pca_scatter(loadings):
    plt.figure(5)
    plt.xlabel('PCA0')
    plt.ylabel('PCA1')
    plt.title('Scatter plot between dimensionally reduced data')
    plt.scatter(loadings['PCA0'], loadings['PCA1'])
    plt.show()


def re_errors(r_errors):
    plt.figure(6)
    plt.xlabel('l (number of dimensions)')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction error for different values of l')
    plt.plot([i for i in range(2, 9)], r_errors)
    plt.show()


# Creating a dataframe for csv file

df = pd.read_csv("pima-indians-diabetes.csv")

# Removing the attribute 'Class'
df.drop('class', inplace=True, axis=1)

df_eigen = df.copy(deep=False)


# Creating a list of names of all attributes
Attributes = ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi', 'Age']


# Replacing outliers with median
for i in Attributes:
    median_calc(i)

# Printing details like mean, std, min, max for data before normalization and standardization
print('Mean and Standard Deviation of original file')
for i in Attributes:
    print(round(df[i].mean(),2), round(df[i].std(),2))
print()

print('Min and Max of original file')
for i in Attributes:
    print(i, min(df[i]), max(df[i]))
print()

# Min Max normalization
for column in Attributes:
    df[column] = (((df[column] - df[column].min()) / (df[column].max() - df[column].min())) * (12 - 5)) + 5

# Printing min max after min - max normalization
print('Min and Max after normalization')
for i in Attributes:
    print(i, min(df[i]), max(df[i]))
print()


# Z standardizaiton
for i in Attributes:
    df[i] = (df[i] - df[i].mean()) / df[i].std()

# Printing mean and std after standardization
print('Mean and Standard deviation after normalization')
for i in Attributes:
    print(round(df[i].mean(),2), round(df[i].std(),2))
print()

# ---------------------------------------------------------------------------------------------

# Details of the synthetic data
mean = [0, 0]
cov = [[13, -3], [-3, 5]]

# Building the synthetic data
x, y = np.random.multivariate_normal(mean, cov, 1000).T

plot_2a(x, y)

# Getting the eigenvalues and eigenvectors for the data
eigen_values, eigen_vectors = np.linalg.eig(cov)

# Initializing origin
origin = [0, 0]

# The two eigenvectors
eig_vec1 = eigen_vectors[:,0]
eig_vec2 = eigen_vectors[:,1]

# Using quiver to plot the eigenvectors.

plot_quiver_line(x, y, eig_vec1, eig_vec2)

# Initializing two lists for storing the data points on each eigenvector
pts = []
pta = []

for i in range(1000):
    ax = x[i]
    ay = y[i]

    # Building the point and line for projection
    point = Point(ax, ay)
    # One of the eigenvectors
    line = LineString([(0, 0), (0.9486833,  -0.31622777)])

    # Projecting each point on a eigenvector
    xa = np.array(point.coords[0])

    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])

    n = v - u
    n /= np.linalg.norm(n, 2)

    P = u + n*np.dot(xa - u, n)

    # Storing the projection of the data point on one eigenvector
    pts.append(P)

for i in range(1000):
    ax = x[i]
    ay = y[i]

    # Building the point and line for projection
    point = Point(ax, ay)

    # The second eigenvector
    line = LineString([(0, 0), (0.31622777, 0.9486833)])

    # Projecting each point on a eigenvector
    xa = np.array(point.coords[0])

    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])

    n = v - u
    n /= np.linalg.norm(n, 2)

    P = u + n*np.dot(xa - u, n)

    # Storing the projection of the data point on one eigenvector
    pta.append(P)


# Diving the projected points into each axis for plotting
n_x = []
n_y = []
for i in range(1000):
    n_x.append(pts[i][0])
    n_y.append(pts[i][1])
plot_3a(x, y, eig_vec1, eig_vec2,n_x, n_y)



# Diving the projected points into each axis for plotting
m_x = []
m_y = []
for i in range(1000):
    m_x.append(pta[i][0])
    m_y.append(pta[i][1])
plot_3b(x, y, eig_vec1, eig_vec2,m_x, m_y)


# Storing the coordinates into two lists for furthur reconstruction
small = []
large = []
for i in range(1000):
    small.append([m_x[i], m_y[i]])
for i in range(1000):
    large.append([n_x[i], n_y[i]])


# Creating a list for calculating reconstruction error
error = []

# Calculating error by calculating euclidean distance between original data and
# Reconstruction data obtained by dot product of the larger eigenvector and the
# data projected on that eigenvector

for i in range(1000):
    error.append(((x[i] - np.dot(large[i], eig_vec1.T))**2  + (y[i] - np.dot(large[i], eig_vec2.T))**2)**0.5)
print('Error calculated manually', sum(error)/len(error))

# Calculating the same error using in-built function by doing pca, and then using
# inverse_transform to get the reconstruction data and then calculating the euclidean
# distance as shown above.

da = pd.DataFrame({'X' : x, 'Y' : y})
pca = PCA(n_components=1)
pca.fit(da)
loadings = pd.DataFrame(pca.transform(da), columns=['PCA%i' % i for i in range(1)], index=da.index)
A = pca.inverse_transform(loadings)
A = A.T
db = pd.DataFrame({'X':A[0], 'Y':A[1]})
s = 0
for i in range(1000):
    a1 = (da['X'].iloc[i] - db['X'].iloc[i]) ** 2
    a2 = (da['Y'].iloc[i] - db['Y'].iloc[i]) ** 2
    s += (a1+a2) ** 0.5
print('Error calculated using PCA', s/1000)
print()

# The difference in reconstruction error between manual and in-built is minimal around 0.2-0.3

# ----------------------------------------------------------------------------------------------

# Value of number of dimensions the data has to be to be converted into
comp = 2

# Using PCA to fit the data
pca = PCA(n_components=comp)
pca.fit(df)

# Storing the l dimensional representation into a dataframe
loadings = pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(comp)], index=df.index)

# Calculating eigenvalue of the dimensionally reduced data
A = np.cov(loadings.T)
w, v = LA.eig(A)

print('Variance of the projected data', pca.explained_variance_)
print('Eigenvalues:', w)

# Plotting the scatter plot of the dimensionally reduced data
pca_scatter(loadings)


# Calculating the values of all the eigenvalues of the original data

eigen_analysis(df_eigen)


# Storing all l values
l_vals = [i for i in range(2, 9)]

# Initializing list for errors
r_errors =[]


for comp in l_vals:
    # Using PCA to fit the data
    pca = PCA(n_components=comp)
    pca.fit(df)

    # Storing the l dimensional representation into a dataframe
    loadings = pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(comp)], index=df.index)
    print('Covariance matrix of l =', comp, 'is', np.cov(loadings.T))
    print()


    # Reconstructing data using inverse_transform
    X_projected = pca.inverse_transform(pca.transform(df))
    da = pd.DataFrame(X_projected, columns = Attributes)

    # Calling function to calculate and return the reconstruction error for each value of l
    r_errors.append(reconstruction_error(df, da, comp))

# Plotting the reconstruction errors
re_errors(r_errors)

print('Covariance matrix of original data', np.cov(df.T))
