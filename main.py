import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading the dataset file
dataset=pd.read_csv("Churn_Modelling.csv")
# Selecting independent and dependant variables
# axis=0 for working on rows and 1 for columns
x=dataset.drop(labels=["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)
y=dataset["Exited"]
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
label_1=LabelEncoder()
x["Geography"]=label_1.fit_transform(x["Geography"])
label_2=LabelEncoder()
x["Gender"]=label_2.fit_transform(x["Gender"])
# Avoiding dummy variable trap (independent variables getting related)
x=pd.get_dummies(x, drop_first=True, columns=["Geography","Gender"])
# Splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(
    x, y, test_size=0.2, random_state=0)
# Feature Scaling (Reducing differences between different features)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# Creating an object (Initializing the ANN)
model=tf.keras.models.Sequential()

# Adding input layer and first hidden layer
# units=6 (trick: avg of input and output layers, here 11+1 /2
# activation function=ReLU, Input dimensions=11
model.add(tf.keras.layers.Dense(units=6, activation="relu", input_dim=11))
# Adding second hidden layer
model.add(tf.keras.layers.Dense(units=6, activation="relu"))
# Output layer: Sigmoid for output of binary classification, softmax for multi
model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
# Compiling model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# Training model (x-np array, y-vector)
model.fit(x_train, y_train.to_numpy(), batch_size=10, epochs=20)


# Evaluate model
test_loss, test_acc= model.evaluate(x_test, y_test.to_numpy())
print(f"Test Accuracy: {test_acc}")
y_pred = (model.predict(x_test) > 0.5).astype ('int32')
print(y_pred)
y_test=y_test.to_numpy()
print(y_pred[0], y_test[0])
print(y_pred[10], y_test[10])
print(y_pred[420], y_test[420])
print(y_pred[120], y_test[120])

# Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test, y_pred)
print(cm)
acc_cm=accuracy_score(y_test, y_pred)
print(acc_cm)
