"""
NAME       - SARTHAK SAPTAMI KUMAR JHA
ROLL NO.   - B20317
MOBILE NO. - 8825319259
BRANCH     - Computer Science and Engineering (CSE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from prettytable import PrettyTable
import scipy as sp
from sklearn.cluster import DBSCAN
from scipy import spatial as spatial


def eigen(df_1):
    plt.figure(1)
    Ax = np.cov(df_1.T)
    w, v = LA.eigh(Ax)
    plt.xlabel('Components')
    plt.ylabel('Eigenvalue')
    plt.title('Magnitude of the eigenvalues')
    plt.plot([1,2,3,4], sorted(w, reverse=True))
    plt.show()
    print('The eigenvalues are :', [round(i, 3) for i in sorted(w, reverse=True)])
    print()


def plt_pca(loadings):
    plt.figure(2)
    plt.title('Reduced data')
    plt.scatter(loadings['PCA0'], loadings['PCA1'])
    plt.show()


def k_cluster(X, y_km):
    plt.figure(3)
    plt.title('K-Means Clustering')
    # Plotting each cluster with different color
    plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=35, c='darkgreen',
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=35, c='darkorange',
    )

    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=35, c='purple',
    )

    # Plotting the centers of the clusters
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=20,
        c='red',
    )
    plt.grid()
    plt.show()


def k_means_distortion_plot(distortions):
    plt.figure(4)
    plt.plot(range(2, 8), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


def gmm_calc(X, y_km):
    plt.figure(5)
    plt.title('GMM Clustering')
    plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=35, c='darkgreen',
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=35, c='darkorange',
    )

    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=35, c='purple',
    )
    centers = np.empty(shape=(gmm.n_components, X.shape[1]))
    for i in range(gmm.n_components):
        density = sp.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(X)
        centers[i, :] = X[np.argmax(density)]
    plt.scatter(centers[:, 0], centers[:, 1],c = 'red', s=20)
    plt.grid()
    plt.show()


def gmm_distortion_plot(distortions):
    plt.figure(7)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Log Likelihood')
    plt.plot([i for i in range(1, 8)], distortions)
    plt.show()


# Function to calculate purity score
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    #print(contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)


# Function for assigning data points to cluster
def dbscan_predict(dbscan_model, X_new):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1
    metric = spatial.distance.euclidean
    # Iterate all input samples for a label
    for j, x_new in (X_new).iterrows():
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new


# Function to perform and plot DBSCAN Results
def dbscanner(loadings, eps, min_samples, k):

    # Figure no.
    plt.figure(k)
    A = loadings.values

    # Creating DBSCAN Model with eps and min_sample values
    # Data fitted is reduced dimensioned data
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples).fit(loadings)

    # Using dbscan_predict function to calculate new labels.
    y_new = dbscan_predict(dbscan_model, loadings)

    # Plotting the data points of each cluster
    for i in list(set(y_new)):
        # Scatter plot for each cluster with random colors
        plt.scatter(A[y_new == i, 0], A[y_new == i, 1], s=25, c=np.random.random(3).reshape(1, -1))

    # Printing number of clusters and noise for each set of values
    print('For eps =', eps, ' and min_samples =', min_samples)
    print('Number of clusters: ', len(list(set(y_new)))-1)
    print('Noise: ', np.count_nonzero(y_new == -1))

    # Printing the above data onto the scatter plot
    textstr = '\n'.join((r'$\mathrm{Clusters}=%.0f$' % (len(list(set(y_new)))-1,),r'$\mathrm{Noise}=%.0f$' % (np.count_nonzero(y_new == -1),)))
    plt.text(1.95, 1.65, textstr, fontsize=12,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.title('Eps = ' + str(eps) + ' Min_Samples = ' + str(min_samples))

    # Returning the purity score
    return purity_score(df['Species'], y_new)


# A function to perform distortion measure manually
def manual_distortion(X, centers):
    s = 0
    for i in range(len(X)):
        a = []
        for j in range(len(centers)):
            a.append(((X[i][0] - centers[j][0]) ** 2) + ((X[i][1] - centers[j][1]) ** 2))
        s += min(a)
    print(s)


# Creating dataframe
df = pd.read_csv("Iris.csv")

# Creating a second dataframe without 'Species' attribute
df_1 = df[["SepalLengthCm", "SepalWidthCm", "SepalWidthCm", "PetalWidthCm"]]
show_plot = 1
pca_plot  = 0


# Q1.

# Performing PCA for 2 components
comp = 2
pca = PCA(n_components=comp)

# Fitting df_1
pca.fit(df_1)

# Creating the reduced dimension dataframe -> loadings
loadings = pd.DataFrame(pca.transform(df_1), columns=['PCA%i' % i for i in range(comp)], index=df_1.index)
A = loadings.values

# Eigen Analysis
if show_plot:
    eigen(df_1)

# Plotting reduced dimension data
if show_plot and pca_plot:
    plt_pca(loadings)


#  Q2.

# Applying K Means clustering on reduced data
K = 3
km = KMeans(n_clusters=K)
km.fit(loadings)
km_pre = km.predict(loadings)

# Plotting the K-Means Clustering
if show_plot:
    k_cluster(A, km_pre)

# Cluster centers of K-Means
centers = km.cluster_centers_

# Calculating Purity Score
print('Purity score for K-Means (K=3):', purity_score(df['Species'], km.labels_))

# Calculating Distortion Measure
manual = 0
if manual:
    manual_distortion(A, centers)
print('Value of distortion measure for K-Means (K=3):', round(km.inertia_, 3))
print()


# Q3.

# lists to store distortion and purity
distortions = []
purity = []

# Looping through 2..8 number of clusters
for i in range(2, 8):
    km = KMeans(n_clusters=i)
    km.fit(loadings)
    km_prea = km.predict(loadings)

    # Calculating purity
    purity.append(purity_score(df['Species'], km_prea))

    # Calculating distortion measure
    distortions.append(km.inertia_)

# Plotting distortion measure for different values of K
if show_plot:
    k_means_distortion_plot(distortions)

# Printing table for Purity Score
myTable = PrettyTable(['K-Value', 'Purity Score'])

for i in range(2, 8):
    myTable.add_row([i, round(purity[i-2], 3)])

print('Purity Scores for K-Means: ')
print(myTable)

# Q4.

# Performing GMM Clustering (K=3)
comp = 3

gmm = GaussianMixture(n_components = comp)
gmm.fit(loadings)
GMM_prediction = gmm.predict(loadings)

# Calculating Purity Score
print()
print('Purity score for GMM(K=3): ', round(purity_score(df['Species'], GMM_prediction), 3))

# Calculating total data log likelihood (distortion measure)
print('Value of distortion measure for GMM Clustering(K=3):',round(gmm.score(loadings), 3))
print()
# Plotting GMM Clusters
gmm_calc(A, GMM_prediction)

# Q5.
# Performing GMM for various number of components

# Creating lists for distortions and purity
distortions = []
purity = []

for i in range(1, 8):
    # Performing GMM
    gmm = GaussianMixture(n_components=i, random_state=42)
    gmm.fit(loadings)

    # Calculating total data log likelihood (distortion measure)
    distortions.append(round(gmm.score(loadings), 3))

    GMM_prediction = gmm.predict(loadings)

    # Calculating Purity score
    purity.append(round(purity_score(df['Species'], GMM_prediction), 3))

gmm_distortion_plot(distortions)

print('Purity score for GMM Clustering:')
myTable = PrettyTable(['K-Value', 'Purity Score'])

for i in range(2, 8):
    myTable.add_row([i, round(purity[i-1], 3)])

print(myTable)
print()

# Q6.

# Performing DBSCAN
k = 8

# List to store Purity Scores
purity = []

# Eps values
for eps in [0.1, 0.5]:
    # Min-Samples values
    for min_samples in [2, 5]:
        purity = (dbscanner(loadings, eps, min_samples, k))
        print('Purity Score:', round(purity, 3))
        print()
        k += 1