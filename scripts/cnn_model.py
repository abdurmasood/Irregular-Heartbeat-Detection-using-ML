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

# classes model needs to learn to classify
CLASSES_TO_CHECK = ['N', 'f']
IMAGES_TO_TRAIN = 100

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

def trainAndTestSplit(df):
    '''
	take dataframe and divide it into train and 
    test data for model training

	Args:
		df (dataframe): dataframe with all images information

	Returns:
		x_train (list): list of training signals
        
        x_test (list): list of testing signals
        
        y_train (list): list of training classes
        
        y_test (list): list of testing classes
	'''
    image_count = 0
    x = []
    y = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for index, row in df.iterrows():
        # check if current row is one of the classes to classify
        if row['Type'] in CLASSES_TO_CHECK:
            x.append(row['Signal'])
            y.append(row['Type'])
            image_count+=1

            # check if   
            if image_count == IMAGES_TO_TRAIN:
                image_count = 0
                CLASSES_TO_CHECK.remove(row['Type'])

        if len(CLASSES_TO_CHECK) == 0:
            break

    # split x and y into train and test data

np.random.seed(1000)

# (2) Get Data
df = getSignalDataFrame()

trainAndTestSplit(df)

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
