import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Activation, Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization

#Data Parameters
num_classes = 2
input_shape = (224, 224, 3)
fixed_seed = 1237213789
batchsize = 32

# download data set from:
# https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign
train_path = './train'
val_path = './test'


# dataset is a tf.data.Dataset object...
# if labels ='none' --> float32 tensors of size (batch_size, img_size[0], img_size[1], num_channels)
# if labels != 'none' --> tuple (images, labels) 
# (images) --> shape is (batch_size, img_size[0], img_size[1], num_channels)
# (labels) --> int32 tensors of shape (batchsize,)

train_dataset = keras.preprocessing.image_dataset_from_directory(
    train_path, # or use /fulldataset
    labels = 'inferred', #this will infer  the classes from 'benign' and 'malignant' folders
    subset = 'training',
    validation_split = 0.2,
    seed = fixed_seed,
    image_size = (224, 224), #it is different from inputshape
)

# # # Merge the 'benign' and 'malignant' folders from /train and /test folders in the /fulldataset folder
# data_path = './fulldataset'
# dataset = keras.preprocessing.image_dataset_from_directory(
#     data_path, # or use /fulldataset
#     labels = 'inferred', #this will infer  the classes from 'benign' and 'malignant' folders
#     subset = 'training',
#     validation_split = 0.2,
#     seed = fixed_seed,
#     image_size = (224, 224), #it is different from inputshape
# )

#print(list(train_dataset.as_numpy_iterator()))
# for data, labels in val_ds:
#     print(data.shape)
#     print(data.dtype)
#     print(labels.shape)
#     print(labels.dtype)

if num_classes == 2:
    dense_activation = 'sigmoid'
    dense_classes = 1
else:
    dense_activation = 'softmax'
    dense_classes = num_classes

# # Model 1 - simple CNN
# model = keras.Sequential(
#     #Input object properties
#     [
#         keras.Input(shape=input_shape), #batch_size=batchsize),
#         layers.experimental.preprocessing.Rescaling(1.0 / 255),
#         #Sequential stacking of layers
#         # Conv2D layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs
#         # filters = 32 --> dim of the outputspace i.e. the number of output filters in the convolution)
#         # kernel_size = 3 --> An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
#         layers.Conv2D(32, 3, padding = 'same', activation = 'relu', kernel_initializer='he_uniform' ),
#         layers.BatchNormalization(),
#         # MaxPooling2D -- > Downsamples the input representation by taking the maximum value over the window defined by pool_size for each dimension along the features axis. The window is shifted by strides in each dimension
#         layers.MaxPooling2D(pool_size = 3, strides = 2, padding = 'same'),

#         # --- next stack of layers
#         layers.Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer='he_uniform' ),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(pool_size = 3, strides = 2, padding = 'same'),

#         layers.Flatten(), #Flattens the input. Does not affect the batch size
#         layers.Dropout(0.5), #The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Note that the Dropout layer only applies when training is set to True such that no values are dropped during inference
#         layers.Dense(dense_classes, activation=dense_activation, kernel_initializer='he_uniform')

#     ]
# )

# # Model 2 - Pre-trained VGG19 model
# from keras.applications.vgg19 import VGG19
# model = keras.Sequential()
# model.add(VGG19(include_top=False, weights='imagenet', input_shape= input_shape))
# for layer in model.layers:
#     layer.trainable=False
# model.add(layers.Flatten())
# model.add(layers.Dense(32))
# model.add(layers.LeakyReLU(0.001))
# model.add(layers.Dense(16))
# model.add(layers.LeakyReLU(0.001))
# model.add(layers.Dense(1, activation='sigmoid'))

# Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015)
# Model 3 - Pre-trained VGG16 model
from keras.applications.vgg16 import VGG16
base_model = VGG16(include_top=False, weights='imagenet', input_shape= input_shape)
for layer in base_model.layers:
    layer.trainable=False
model = keras.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


model.summary()
epochs = 2
model.compile(loss = 'binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #worse performance

#model.fit(dataset, epochs=epochs)
model.fit(train_dataset, epochs=epochs)

wait = input("Please Press Enter")
#score = model.evaluate(dataset)
# --- Loading and evaluating validation data
val_dataset = keras.preprocessing.image_dataset_from_directory(
    val_path,
    labels = 'inferred',
    # subset = 'validation',
    # validation_split = 1,
    seed = fixed_seed,
    image_size = (224, 224), #it is different from inputshape
)


score = model.evaluate(val_dataset)
print("Test loss:", score[0])
print("Test accuracy:", score[1])