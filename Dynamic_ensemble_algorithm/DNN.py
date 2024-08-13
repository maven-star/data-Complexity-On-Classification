import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def Classify(Data,Label,tr):


    x_train, x_test, y_train, y_test = train_test_split(Data, Label, train_size=tr, random_state=0) # 70% for train and 30% for test
    nc=len(np.unique(y_train))
    # Defining Model
    # Using Sequential() to build layers one after another
    model = tf.keras.Sequential([

        # Flatten Layer that converts images to 1D array
        tf.keras.layers.Flatten(),

        # Hidden Layer with 512 units and relu activation
        tf.keras.layers.Dense(units=512, activation='relu'),

        # Output Layer with 10 units for 10 classes and softmax activation
        tf.keras.layers.Dense(units=nc, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )


    # Making Predictions
    predict = model.predict(np.array(x_test))

    return predict[:,0],y_test





