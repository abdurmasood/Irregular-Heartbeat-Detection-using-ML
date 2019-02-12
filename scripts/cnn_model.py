# Implimentation of AlexNet model taken from https://www.mydatahack.com/building-alexnet-with-keras/
# script that reads data, creates model and trains it

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import directory_structure
import os
import cv2
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import cifar10

# classes model needs to learn to classify
CLASSES_TO_CHECK = ['N', 'f', 'A']
IMAGES_TO_TRAIN = 0

# removing warning for tensorflow about AVX support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def getSignalDataFrame():
    '''
	read signal images present in the directory beat_write_dir
    and save them in a dataframe

	Returns:
		(dataframe): dataframe contatining image information 
	'''
     #get paths for where signals are present
    signal_path = directory_structure.getWriteDirectory('beat_write_dir', None)

    #create dataframe
    df = pd.DataFrame(columns=['Signal ID', 'Signal', 'Type'])

    arrhythmia_classes = directory_structure.getAllSubfoldersOfFolder(signal_path)

    image_paths = []
    images = []
    image_ids = []
    class_types = []

    #get path for each image in classification folders
    for classification in arrhythmia_classes:
        classification_path = ''.join([signal_path, classification])
        image_list = directory_structure.filesInDirectory('.png', classification_path)
        for beat_id in image_list:
            image_ids.append(directory_structure.removeFileExtension(beat_id))
            class_types.append(classification)
            image_paths.append(''.join([classification_path, '/' ,beat_id]))

    # read and save images in dataframe
    for path in image_paths:
        img =cv2.imread(path)
        images.append(img)

    # save information in dataframe
    df['Signal ID'] = image_ids
    df['Type'] = class_types
    df['Signal'] = images

    return df

def normalizeData(X_train, X_test, y_train, y_test):
    '''
    Normalizing the test and train data
    '''
    num_of_classes = len(CLASSES_TO_CHECK) + 1

    # image normalization
    X_train = X_train.astype('float32')
    X_train = X_train / 255
    X_test = X_test.astype('float32')
    X_test = X_test / 255

    # label normalization
    y_train = keras.utils.to_categorical(y_train, num_of_classes)
    y_test = keras.utils.to_categorical(y_test, num_of_classes)

    return X_train, X_test, y_train, y_test

def convertToNumpy(X_train, X_test, y_train, y_test):
    '''
    Convert data arrays into numpy arrays
    '''
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def trainAndTestSplit(df, images_to_train, size_of_test_data):
    '''
	take dataframe and divide it into train and 
    test data for model training

	Args:
		df (dataframe): dataframe with all images information

        images_to_train (int): number of images to get for training
                               from dataframe
        
        size_of_test_data (float): percentage of data specified for training

	Returns:
		X_train (list): list of training signals
        
        X_test (list): list of testing signals
        
        y_train (list): list of training classes
        
        y_test (list): list of testing classes
	'''

    IMAGES_TO_TRAIN = images_to_train
    image_count = 0

    # train + test data (signals and classes of signals respectively)
    X = []
    y = []

    for index, row in df.iterrows():
        # check if current row is one of the classes to classify
        if row['Type'] in CLASSES_TO_CHECK:
            X.append(row['Signal'])
            y.append(CLASSES_TO_CHECK.index(row['Type']))
            image_count+=1

            if image_count == IMAGES_TO_TRAIN:
                image_count = 0
                CLASSES_TO_CHECK.remove(row['Type'])

        # if data collected from all classes break loop
        if len(CLASSES_TO_CHECK) == 0:
            break

    # split x and y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size_of_test_data)

    # convert to numpy array
    X_train, X_test, y_train, y_test = convertToNumpy(X_train, X_test, y_train, y_test)

    # normalize data for easy data processing
    X_train, X_test, y_train, y_test = normalizeData(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test

# (2) GET DATA
df = getSignalDataFrame()

X_train, X_test, y_train, y_test = trainAndTestSplit(df, 100, 0.2)

# (3) CREATE SEQUENTIAL MODEL
model = Sequential()

# -----------------------1st Convolutional Layer--------------------------
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
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
model.add(Dense(3, activation='softmax'))

# uncomment to print out summary of model
model.summary()

# (4) COMPILE MODEL
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

# (5) TRAIN
model.fit(
    X_train, 
    y_train, 
    batch_size=64, 
    epochs=60, 
    verbose=1,
    validation_data=(X_test, y_test),
    shuffle=True
)
