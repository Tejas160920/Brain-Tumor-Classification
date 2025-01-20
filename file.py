# Two types. 1. Predict user based on eeg data and class of hand movement. 
# 2. Predict hand movement based on user.

# plotting is 
# x = input values from the electrodes
# y = user data or class data



import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

dfs = [pd.read_csv('E:\\3RD YEAR SEM-2\\EDI\\Dataset\\user_'+ user + '.csv') for user in ['a', 'b', 'c', 'd']]
# print(dfs[3])

for i in range(len(dfs)):
    dfs[i]['User'] = pd.Series(i, index = dfs[i].index)

data = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True) #to join all dataasets and shuffle all elements randomly  
# reset_index to reset index ia a particular order and drop = true for preventing new column to be formed.
# print(data)


def onehot_encode(df, column):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

onehot_encode( data, column='Class')


def preprocess_inputs(df, target = 'Class'):
    df = df.copy()

    # One-hot encode whichever target column is not being used
    targets = ['Class', 'User']
    targets.remove(target)
    df = onehot_encode(df, column=targets[0])

    #  Split the df into x and y where x will be everything except the target column

    y = df[target].copy()
    x = df.drop(target, axis=1)

    #train_test_split(to train some data and test some data)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.7, random_state=123) # returns 4 values# 70 percent data is being taken to train and this function by default shuffles the data
    
    # Scaling X with a standard scalar to select a particular range of values

    scaler = StandardScaler()
    scaler.fit(X_train) # we scale only x since x is our input 

    X_train = pd.DataFrame(scaler.transform(X_train), columns = x.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = x.columns) # transform function returns numpy array so we use dataframe to convert it into dataframe
    
    return X_train, X_test, Y_train, Y_test; # columns have been scaled so that the mean of each column is very close to zero and variance is very close to one because of which we can say they all take on the same range of values


def build_model(num_classes = 3):
    
    inputs = tf.keras.Input(shape=[X_train.shape[1], ])
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(
        optimizer='adam',    #optimize the input weights by comparing the prediction and the loss function.
        loss='sparse_categorical_crossentropy', # to calculate the loss function i.e to find any error or deviation
        metrics=['accuracy']     #calculate accuracy
    )

    return model


# For prediction of hand movement done by user

print("\n\nFor prediction of hand movement done by user \n\n")

X_train, X_test, Y_train, Y_test = preprocess_inputs(data, target='Class')

class_model = build_model(num_classes=3)

class_history = class_model.fit(

    X_train,
    Y_train,
    validation_split = 0.2,
    batch_size = 32,
    epochs = 50,
    callbacks=[
    
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    ]
)


class_acc = class_model.evaluate(X_test, Y_test, verbose=0)[1] #since we want only accuracy column
print("Test Accuracy (Class Model): {:.2f}%".format(class_acc*100))




# print(X_train)

# a = X_train.var()
# print(a)

# b = Y_train.value_counts()
# print(b)

# c = X_train.shape(1) #a total of 115 values are recorded from these that means 115 columns == 115 features for each row
# print( c )


# For prediction of User on basis of hand movement

print("\n\n\nFor prediction of User on basis of hand movement\n\n")
X_train, X_test, Y_train, Y_test = preprocess_inputs(data, target='User')

User_model = build_model(num_classes=4)

User_history = User_model.fit(

    X_train,
    Y_train,
    validation_split = 0.2,
    batch_size = 32,
    epochs = 50,
    callbacks=[
    
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    ]
)

User_acc = User_model.evaluate(X_test, Y_test, verbose=0)[1] #since we want only accuracy column
print("Test Accuracy (User Model): {:.2f}%".format(User_acc*100))