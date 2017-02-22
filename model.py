import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 10, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

DATA_DIR = './data/'

# Read data
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv(DATA_DIR + 'driving_log.csv', skiprows=1, names=column_names)
center = data.center.tolist()
left = data.left.tolist()
right = data.right.tolist()
steering = data.steering.tolist()

center, steering = shuffle(center, steering)
center, X_valid, steering, y_valid = train_test_split(center, steering, test_size=0.1)


def preprocess_image(image):
    return cv2.resize(image[50:140, :], (80, 80))


def generator(X, y, batch_size=32):
    num_samples = len(X)
    X, y = shuffle(X, y)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]

            images = []
            angles = []
            for i in range(len(batch_X)):
                name = DATA_DIR + 'IMG/'+ X[i].split('/')[-1]
                original_image = cv2.imread(name)
                center_image = preprocess_image(original_image)
                center_angle = float(batch_y[i])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(center, steering, batch_size=32)
validation_generator = generator(X_valid, y_valid, batch_size=32)

if __name__ == "__main__":
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(80, 80, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2,2), W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(40, W_regularizer=l2(0.001)))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dense(1, W_regularizer=l2(0.001)))
    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
    model.summary()

    model.fit_generator(train_generator, validation_data=validation_generator,
                        nb_epoch=FLAGS.epochs, samples_per_epoch=len(center),
                        nb_val_samples=len(X_valid))

    model.save('model.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)