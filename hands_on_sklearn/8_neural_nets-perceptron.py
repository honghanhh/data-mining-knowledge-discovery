""" ---------------------------------------------------------------"""
"""          Neural networks introduction              """
""" ---------------------------------------------------------------"""
import pandas as pd
from sklearn.model_selection import train_test_split
#import fix_random_seed


# -------------------------------------------------------------------------------
# For loading the data, the code is practically the same as in 2_classification.py
print(""" --- Load the data ---""")
csvFileName = r"./Datasets/A-greater-then-B.csv "
df = pd.read_csv(csvFileName)
print(df.head())
print("data shape: ", df.shape)

print(""" --- Set the features (independednt variables, attributes) and target ---""")
feature_cols = ['A', 'B', 'C']
target_var = 'A>B'

print( """ --- transform from categorical target (True, False) into numeric (1, 0) --- """)
df[target_var] = df[target_var].map(lambda x: 1 if x==True else 0)
print(df.head())

X = df[feature_cols].values
y = df[target_var].values
print("Features: ", feature_cols, "\nTarget:", target_var)

print(""" --- Train-test split ---""")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("train set X shape: ", X_train.shape, "train set y shape: ", y_train.shape)
print("test set X shape: ", X_test.shape, "test set y shape: ", y_test.shape)

# -------------------------------------------------------------------------------
print (""" --- Introducing a validation set --- """)
print(""" --- Train-validation split ---""")
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print("train set X shape: ", X_train.shape, "train set y shape: ", y_train.shape)
print("validation set X shape: ", X_validation.shape, "validation set y shape: ", y_validation.shape)
print("test set X shape: ", X_test.shape, "test set y shape: ", y_test.shape)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
print("""-----------------------------------------------------""")
print("""\n ---- A Single Perceptron or Shallow network --- \n""")

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(input_dim=3, output_dim=1, init='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=10, batch_size=64, verbose=0)

print(""" --- Predict --- (10 examples from the test set)""")
y_pred = model.predict(X_test)
print(" Actual   Predicted   Difference")
#for i in range(len(y_test)):
for i in range(10):
    print("{0:6.2f}  {1:8.2f}    {2:8.2f}".format(y_test[i], y_pred[i][0], y_test[i]- y_pred[i][0]))

# Model perfomance
print(""" -- Model Performance ---""")
# Returns the loss value & metrics values for the model in test mode.
scores = model.evaluate(X_train, y_train, verbose=0)
print("Train set error: ", scores)

scores = model.evaluate(X_validation, y_validation, verbose=0)
print("Validation set error: ", scores)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Test set error: ", scores)
