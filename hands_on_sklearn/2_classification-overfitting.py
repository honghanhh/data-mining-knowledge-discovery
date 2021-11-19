import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

print("_______________________________________________________________________________")
print("            Classification, fitting & overfitting. ")
print("_______________________________________________________________________________")

print(""" --- Load the data ---""")
csvFileName = r"./Datasets/A-greater-then-B.csv "
df = pd.read_csv(csvFileName)
print(df.head())
print("data shape: ", df.shape)

print(""" --- Set the features (independednt variables, attributes) and target ---""")
feature_cols = ['A', 'B', 'C']
target_var = 'A>B'

X = df[feature_cols].values
y = df[target_var].values
print("Features: ", feature_cols, "\nTarget:", target_var)

print(""" --- Train-test split ---""")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("train set X shape: ", X_train.shape, "train set y shape: ", y_train.shape)
print("test set X shape: ", X_test.shape, "test set y shape: ", y_test.shape)


print("decision_tree.tree_.node_count\t acc(train)\t acc(test)")
for i in range(2, 100):

    decision_tree = tree.DecisionTreeClassifier(max_leaf_nodes=i)
    decision_tree.fit(X_train, y_train)

    y_pred_test = decision_tree.predict(X_test)
    y_pred_train = decision_tree.predict(X_train)

    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    acc_test = metrics.accuracy_score(y_test, y_pred_test)

    print("\t{0:5.2f}\t{1:5.2f}\t{2:5.2f}".format(decision_tree.tree_.node_count, acc_train,acc_test ))

