import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

print("_______________________________________________________________________________")
print("            Classification, fitting, predicting ... printing. ")
print("_______________________________________________________________________________")

print(""" --- Load the data ---""")
csvFileName = r"./Datasets/A-greater-then-B.csv "
df = pd.read_csv(csvFileName)
print(df.head())
print("data shape: ", df.shape)

print(""" --- Set the features (independent variables, attributes) and target ---""")
feature_cols = ['A', 'B', 'C']
target_var = 'A>B'

X = df[feature_cols].values
y = df[target_var].values
print("Features: ", feature_cols, "\nTarget:", target_var)

print(""" --- Train-test split ---""")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("train set X shape: ", X_train.shape, "train set y shape: ", y_train.shape)
print("test set X shape: ", X_test.shape, "test set y shape: ", y_test.shape)

print(""" --- Initialize the learner(s) --- """)
# documentation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
decision_tree = tree.DecisionTreeClassifier()
print("Parameters of the decision tree: ", decision_tree.get_params())

print(""" --- Fit --- """)
decision_tree.fit(X_train, y_train)
print("Depth of the decision tree: ", decision_tree.tree_.max_depth)
print("Number of nodes of the decision tree: ", decision_tree.tree_.node_count)

print(""" --- Predict --- """)
y_pred = decision_tree.predict(X_test)
print("\n Actual   Predicted")
#for i in range(len(y_test)):
for i in range(10):
    print("{0:6.2f}  {1:8.2f}".format(y_test[i], y_pred[i]))
print(""" -- Performance ---""")

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy  \t{0:5.2f}".format(accuracy))


""" Visualize the tree """               # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
treeFileName = 'decision_tree.dot'
tree.export_graphviz(decision_tree, out_file=treeFileName)
#
# # install GraphViz
#$ dot -Tps decision_tree.dot -o decision_tree.ps      (PostScript format)
# # $ dot -Tpng decision_tree.dot -o decision_tree.png    (PNG format)
