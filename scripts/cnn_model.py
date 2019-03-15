# Implmentation of AlexNet model taken from https://www.mydatahack.com/building-alexnet-with-keras/
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
from sklearn.model_selection import train_test_split
import keras
from collections import deque
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping

# classes model needs to learn to classify
CLASSES_TO_CHECK = [ 'A', 'L', 'N']
NUMBER_OF_CLASSES = len(CLASSES_TO_CHECK)
IMAGES_TO_TRAIN = 50

# removing warning for tensorflow about AVX support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def saveMetricsAndWeights(score, model):
    loss = score[0]
    current_acc = score[1]

    directory_structure.getWriteDirectory('testing', None)
    weights_path = directory_structure.getWriteDirectory('testing', 'model_weights')
    metrics_path = directory_structure.getWriteDirectory('testing', 'accuracy_metrics')

    if (len(directory_structure.filesInDirectory('.npy', metrics_path)) == 0):
        # create text file with placeholder accuracy value (i.e 0)
        np.save(metrics_path + 'metrics.npy', [0])
        model.save(weights_path + 'my_model.h5')
        del model
    else:
        highest_acc = np.load(metrics_path + 'metrics.npy')[0]
        if (current_acc > highest_acc):
            np.save(metrics_path + 'metrics.npy', [current_acc])
            model.save(weights_path + 'my_model.h5')
            del model
            print('\nAccuracy Increase: ' + str(current_acc - highest_acc) + '%')


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

    image_paths = deque()
    image_ids = deque()
    class_types = deque()
    images = []

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
        images.append(cv2.imread(path))

    # save information in dataframe
    df['Signal ID'] = image_ids
    df['Type'] = class_types
    df['Signal'] = images

    return df

def normalizeData(X_train, X_test, y_train, y_test):
    '''
    Normalizing the test and train data
    '''

    # image normalization
    X_train = X_train.astype('float32')
    X_train = X_train / 255
    X_test = X_test.astype('float32')
    X_test = X_test / 255

    # label normalization
    y_train = keras.utils.to_categorical(y_train, NUMBER_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUMBER_OF_CLASSES)

    return X_train, X_test, y_train, y_test

def convertToNumpy(X_train, X_test, y_train, y_test):
    '''
    Convert data arrays into numpy arrays
    '''
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def trainAndTestSplit(df, size_of_test_data):
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
    image_count = 0
    classes_to_check = CLASSES_TO_CHECK

    # train + test data (signals and classes of signals respectively)
    X = []
    y = []

    for index, row in df.iterrows():
        # check if current row is one of the classes to classify
        if row['Type'] in classes_to_check:
            X.append(row['Signal'])
            y.append(classes_to_check.index(row['Type']))
            image_count+=1

            if image_count == IMAGES_TO_TRAIN:
                image_count = 0
                classes_to_check.remove(row['Type'])

        # if data collected from all classes break loop
        if len(classes_to_check) == 0:
            break

    # split x and y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size_of_test_data)

    # convert to numpy array
    X_train, X_test, y_train, y_test = convertToNumpy(X_train, X_test, y_train, y_test)

    # normalize data for easy data processing
    X_train, X_test, y_train, y_test = normalizeData(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test

def printTestMetrics(score):
    '''
    print prediction score

    Args:
        score (list): list with test loss and test accuracy
    '''
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def createModel(model_name):
    '''
    Implementation of model to train images (Alexnet or Novelnet)

    Args:
        model_name (str): name of the model to create (can choose from Alexnet and Novelnet)

    Returns:
        model (model): model object implementation of alexnet
    '''
    model = Sequential()

    if model_name == 'Alexnet':
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
        model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    
    elif model_name == 'Novelnet':
        # -----------------------1st Convolutional Layer--------------------------
        model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(13,13),\
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
        model.add(Conv2D(filters=434, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -----------------------5th Convolutional Layer----------------------------
        model.add(Conv2D(filters=500, kernel_size=(3,3), strides=(1,1), padding='valid'))
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
        model.add(Dropout(0.6))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -------------------------3rd Dense Layer---------------------------
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.7))
        # Batch Normalisation
        model.add(BatchNormalization())

        # --------------------------Output Layer-----------------------------
        model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    

    return model

if __name__ == '__main__':

    # (2) GET DATA
    df = getSignalDataFrame()

    X_train, X_test, y_train, y_test = trainAndTestSplit(df, 0.15)

    # (3) CREATE SEQUENTIAL MODEL
    model = createModel('Alexnet')

   # uncomment to do computation on multiple gpus
    #parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model = model

    # (4) COMPILE MODEL
    parallel_model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    # (5) TRAIN
    history = parallel_model.fit(
        X_train, 
        y_train, 
        batch_size=64, 
        epochs=30, 
        verbose=1,
        validation_data=(X_test, y_test),
        shuffle=True,
    )

    # (6) PREDICTION
    predictions = parallel_model.predict(X_test)
    score = parallel_model.evaluate(X_test, y_test, verbose=0)

    printTestMetrics(score)

    # (7) SAVE TESTS + WEIGHTS 
    saveMetricsAndWeights(score, parallel_model)