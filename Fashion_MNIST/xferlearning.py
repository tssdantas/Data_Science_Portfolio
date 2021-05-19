import numpy as np
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras import Input as keras_Input
from keras import layers
from keras.models import Sequential
from keras.utils import np_utils
from numpy.core.fromnumeric import diagonal
from sklearn import metrics

fixed_seed = 4328794
np.random.seed(fixed_seed)


#loading data sets
imgsize = (28, 28) # Both data sets feature same image sizes!
inputsize = imgsize + (1,) #Grayscale image ... alternative: use np.expand_dims(X_train, -1) 

(X_train1, y_train1), (X_test1, y_test1) = fashion_mnist.load_data()
(X_train2, y_train2), (X_test2, y_test2) = mnist.load_data()

#Preprocessing
X_train1.astype('float32')
X_train2.astype('float32')
X_test1.astype('float32')
X_test2.astype('float32')

#CNN Params
nr_Conv_filters = 32
kernel_size = 3
pool_size = 2

#Training parameters
epochs = 20
nr_classes = 10

#Stacking layers of the Sequetial CNN
features_layers = [
    keras_Input(shape=inputsize),
    layers.experimental.preprocessing.Rescaling(1.0 / 255),
    layers.Conv2D(nr_Conv_filters, kernel_size, padding='valid', activation='relu'),#, input_shape=inputsize),
    layers.Conv2D(2*nr_Conv_filters, kernel_size, activation='relu'),
    layers.MaxPooling2D(pool_size=(pool_size, pool_size)),
    layers.Dropout(0.25),
    layers.Flatten()
]
classification_layers = [
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(nr_classes, activation= 'softmax'),
]


def train_model(model, nr_classes_in, epochs_in, X_train, y_train, X_test, y_test):
    #Preprocessing Output
    y_train1_cat = np_utils.to_categorical(y_train, nr_classes_in)
    y_test1_cat = np_utils.to_categorical(y_test, nr_classes_in)

    #model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train1_cat, epochs=epochs_in, verbose=1, validation_data = (X_test, y_test1_cat))
    score = model.evaluate(X_test, y_test1_cat, verbose = 0)
    print('Test score : ', score[0])
    print('Test accuracy: ', score[1])

    y_train_predicted = model.predict(X_train)
    confusion_matrix = metrics.confusion_matrix(y_train, np.argmax(y_train_predicted, axis=1), labels=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9. ])
    
    print('\nConfussion Matrix : ', confusion_matrix)

    #Estimating accuracy of classification of each class
    accuracy_per_class = np.zeros((1, 10), dtype='float32')
    temp_sum = confusion_matrix.sum(axis=1)
    diagonal_vec = np.diag(confusion_matrix)
    it = np.nditer(diagonal_vec, flags=['f_index'])
    for value in it:
        accuracy_per_class[0,it.index] = value / temp_sum[it.index]
    # Fashion MNIST classes Ref: https://keras.io/api/datasets/fashion_mnist/
    print('\nAccuracy of prediction per class\n', accuracy_per_class)

model = Sequential(features_layers + classification_layers )

train_model(model, nr_classes, epochs, X_train2, y_train2, X_test2, y_test2 )

#freeze features layers and rebuild model
for layer in features_layers:
    layer.trainable = False

#Now retraining the model, with flexible classification layers and fixed feature layers from MNIST training
train_model(model, nr_classes, epochs, X_train1, y_train1, X_test1, y_test1 )
