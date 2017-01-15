from __future__ import print_function
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as k

np.random.seed(1337)  # for reproducibility

MAX_ROW = 60
MAX_COL = 100

class ConvolutionNeuralNetwork:
    def __init__(self, data, label):
        self.training_model = None
        self.batch_size = 128
        self.nb_classes = 2
        self.nb_epoch = 1

        # input image dimensions
        self.img_rows = MAX_ROW
        self.img_cols = MAX_COL

        # number of convolutions filters to use
        self.nb_filters = 32

        # size of pooling area for max pooling
        self.pool_size = (2, 2)

        # convolution kernel size
        self.kernel_size = (3, 3)

        # Training dataset
        self.data = data
        self.label = label

    def resize_and_pad(self, img):
        row, col, dep = img.shape
        if row > self.img_rows or col > self.img_cols:
            x = float(self.img_rows) / row
            y = float(self.img_cols) / col
            factor = min(x, y)
            img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            row, col, dep = img.shape
        top = bottom = (self.img_rows - row) / 2
        left = right = (self.img_cols - col) / 2
        des = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=1)
        return des

    def preprocess(self, X_data, y_data=None):
        # Depth specify the pixel dimensionality i.e. 3 for RGB and 1 for grey scale.
        depth = 3

        if k.image_dim_ordering() == 'th':
            X_data = X_data.reshape(X_data.shape[0], depth, self.img_rows, self.img_cols)
            self.input_shape = (depth, self.img_rows, self.img_cols)
        else:
            X_data = X_data.reshape(X_data.shape[0], self.img_rows, self.img_cols, depth)
            self.input_shape = (self.img_rows, self.img_cols, depth)

        X_data = X_data.astype('float32')
        X_data /= 255

        print('X_data shape:', X_data.shape)
        print(X_data.shape[0], 'train samples')

        # convert class vectors to binary class matrices
        if y_data is not None:
            y_data = np_utils.to_categorical(y_data, self.nb_classes)
            print('label shape', y_data.shape)
            return X_data, y_data
        return X_data

    def define_model(self):
        model = Sequential()
        model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1], border_mode='valid',
                                input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        return model

    def cross_validate(self):
        self.data, self.label = self.preprocess(self.data, self.label)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.label, test_size=0.4,
                                                                                random_state=7, stratify=self.label)
        model = self.define_model()
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        model.fit(self.X_train, self.y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=1,
                  validation_data=(self.X_test, self.y_test))
        score = model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def train_model(self):
        self.training_model = self.define_model()
        self.training_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.training_model.fit(self.data, self.label, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=0)

    def make_prediction(self, test_data):
        test_data = self.preprocess(test_data)
        predictions = self.training_model.predict_classes(test_data, batch_size=self.batch_size, verbose=0)

        return predictions
# End of module
