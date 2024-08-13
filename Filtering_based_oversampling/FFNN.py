from tensorflow import keras
from sklearn.model_selection import train_test_split
import  numpy as np


def classify(Data,Label,tr):
    x_train, x_test, y_train, y_test = train_test_split(Data, Label, train_size=tr, random_state=0) # 70% for train and 30% for test

    x_train=np.resize(x_train,(300,300))
    x_test = np.resize(x_test, (300,300))
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_test=np.resize(y_test,len(x_test))

    model = keras.Sequential()
    a, b = 12, 5

    # Choose an optimal value between 10-300

    model.add(keras.layers.Dense(units=512, input_shape=x_train.shape[1:], activation='relu'))

    model.add(keras.layers.Dense(units=10, activation='relu' ))

    model.add(keras.layers.Dense(units=10, activation='relu' ))

    model.add(keras.layers.Dense(units=10, activation='relu' ))

    model.add(keras.layers.Dense(units=10, activation='relu'))



    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
   # tf.keras.utils.plot_model(model, to_file='DFFNN.png', show_shapes=True, show_layer_names=True)

    y_train=np.resize(y_train,len(y_test))
    model.fit(x_train, y_train, epochs=10, batch_size=10,validation_data=(x_test, y_test), verbose=0)

    pred = model.predict(x_test)
    return pred[:,0],y_test

