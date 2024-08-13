import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split  # Import train_test_split function
import numpy as np,math



def Classify(Data,Label,tr):

    X_train, X_test, y_train, y_test = train_test_split(Data, Label, train_size=tr, random_state=42)


    num_classes = len(np.unique(y_train))


    xt=len(X_test)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(num_classes, activation='softmax'))
    X_test = [X_test[i][0:100] for i in range(len(X_test))]
    X_test = np.resize(X_test, (xt, 28, 28, 1))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam',
                          metrics=['accuracy'])
    pred = model.predict(X_test)

    return pred[:,0],y_test