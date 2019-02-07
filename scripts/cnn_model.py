# Code for AlexNet model taken from https://www.mydatahack.com/building-alexnet-with-keras/

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import directory_structure
import os

np.random.seed(1000)

# removing error for tensorflow about AVX support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# (2) Get Data


# (3) CREATE SEQUENTIAL MODEL
model = Sequential()

# -----------------------1st Convolutional Layer--------------------------
model.add(Conv2D(filters=96, input_shape=(224,224,1), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())


# -----------------------2nd Convolutional Layer---------------------------
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# -----------------------3rd Convolutional Layer----------------------------
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# -----------------------4th Convolutional Layer----------------------------
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# -----------------------5th Convolutional Layer----------------------------
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# -------------------------1st Dense Layer----------------------------
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# -------------------------2nd Dense Layer---------------------------
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# -------------------------3rd Dense Layer---------------------------
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# --------------------------Output Layer-----------------------------
model.add(Dense(17))
model.add(Activation('softmax'))

# uncomment to print out summary of model
# model.summary()

# (4) Compile 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # (5) Train
# model.fit(x, y, batch_size=64, epochs=1, verbose=1, \
# validation_split=0.2, shuffle=True)