

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time

""" --------------------------------"""
"""          Clustering             """
""" --------------------------------"""
# https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html

""" Load and plot the data """
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

data = np.load('./Datasets/clusterable_data.npy')
print(data.shape)


f = plt.figure(1)
plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)


def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.55, 0.65, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)



f2 = plt.figure(2)
plot_clusters(data, cluster.KMeans, (), {'n_clusters':6})

# add a plot with a different number of clusters

f3 = plt.figure(3)
plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})

f4 = plt.figure(4)
plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters':6})

f5 = plt.figure(5)
plot_clusters(data, cluster.DBSCAN, (), {'eps':0.025})

plt.show()