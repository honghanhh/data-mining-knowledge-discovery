import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import metrics

n_neighbors = 15
#------------------------------------------------------------------------------------
print(""" --- Load the data ---""")
csvFileName = r"./Datasets/A-greater-then-B.csv "
df = pd.read_csv(csvFileName)
print(df.head())
print("data shape: ", df.shape)

print(""" --- Set the features (independent variables, attributes) and target ---""")
feature_cols = ['A', 'B']   # we don't want the random column C for this experiment
target_var = 'A>B'

X = df[feature_cols].values
y = df[target_var].values
print("Features: ", feature_cols, "\nTarget:", target_var)

print(""" --- Train-test split ---""") # 900 examples in test set, 100 examples in train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=900, random_state=2)
h = .001  # step size in the mesh

# adding one noisy example
X_train = np.vstack((X_train, [0.25, 0.75]))
y_train = np.append(y_train, y_train[0])



# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

for n_neighbors in [1, 3, 10, 30]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)


    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    xx, yy = np.meshgrid(np.arange(0, 1, h), np.arange(0, 1, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='face', s=10)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=50)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("A>B classification (k = %i), accuracy = %5.2f" % (n_neighbors, accuracy))



plt.show()