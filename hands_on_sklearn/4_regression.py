import pandas as pd
from sklearn import dummy
from sklearn import linear_model
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

print("_______________________________________________________________________________")
print("Regression models, train-test validation on regressionAgeHeight.csv. ")
print("_______________________________________________________________________________")

print(""" Load the data """)
csvFileName = r"./Datasets/regressionAgeHeight.csv"
df = pd.read_csv(csvFileName)
print(df.head())
print("data shape: ", df.shape)

feature_cols = ['Age']
target_var = 'Height'

X = df[feature_cols].values
y = df[target_var].values

""" Train-test split """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

""" Initialize the learners """
dummy = dummy.DummyRegressor()
regr = linear_model.LinearRegression()
reg_tree = tree.DecisionTreeRegressor(min_samples_leaf=8)
knn = KNeighborsRegressor(n_neighbors=2)

learner = reg_tree

"""" Train and apply """
learner.fit(X_train, y_train)
y_pred = learner.predict(X_test)

print ("\n Actual   Predicted")
for i in range(len(y_test)):
     print("{0:6.2f}  {1:8.2f}".format(y_test[i], y_pred[i]))

print("Performance:")
print("MAE  \t{0:5.2f}".format( metrics.mean_absolute_error(y_test,y_pred)))
print("MSE  \t{0:5.2f}".format( metrics.mean_squared_error(y_test,y_pred)))
print("R2   \t{0:5.2f}".format( metrics.r2_score(y_test,y_pred)))
#
# # """ Visualize the tree """               # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
treeFileName = 'reg_tree.dot'
tree.export_graphviz(learner, out_file='treeFileName.dot')
#
# # install GraphViz
#$ dot -Tps reg_tree.dot -o tree.ps      (PostScript format)
# # $ dot -Tpng reg_tree.dot -o tree.png    (PNG format)
