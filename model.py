###################
# IMPORT THE DATA #
###################

import csv
import cv2
import numpy as np

### Read the driving log
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

### Grab the images and steering inputs for every frame        
images, measurements, augmented_images, augmented_measurements = [], [], [], []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        ### Add a constant steering offset for the left and right cameras
        if i == 0: ### center camera
            offset = 0
        if i == 1: ### left camera
            offset = 0.2
        if i == 2: ### right camera
            offset = -0.2
        measurement = float(line[3]) + offset
        measurements.append(measurement)

### Create an augmented data set from mirrored data        
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1)) ### flip the image
    augmented_measurements.append(measurement*-1.0) ### flip the steering angle

### Define the training/validation data
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

########################
# DEFINE NETWORK MODEL #
########################

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D

### NVIDIA End-to-End CNN (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

############################
# OPTIMIZE AND TRAIN MODEL #
############################

### Using MSE loss function and Adam optimizer
model.compile(loss = 'mse', optimizer = 'adam')

### Train and save the model
history_object = model.fit(X_train, y_train, validation_split = 0.1, shuffle = True, nb_epoch = 5)
model.save('model.h5')

#############
# PLOT LOSS #
#############

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()