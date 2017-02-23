import random

import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
tf.python.control_flow_ops = tf


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")
LEFT_PERCENT = 1.0
RIGHT_PERCENT = 1.0
DATA_DIR = './data/'

# Read data
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv(DATA_DIR + 'driving_log.csv', skiprows=1, names=column_names)
center = data.center.tolist()
left = data.left.tolist()
right = data.right.tolist()
steering = data.steering.tolist()

center, left, right, steering = shuffle(center, left, right, steering)

print(len(steering))

limit = 100
steering_ids = []
for i, x in enumerate(steering):
    if abs(x) > 0.1:
        steering_ids.append(i)
    elif limit > 0:
        steering_ids.append(i)
        limit -= 1

print(len(steering_ids))
steering = [steering[i] for i in steering_ids]
left = [left[i] for i in steering_ids]
right = [right[i] for i in steering_ids]
center = [center[i] for i in steering_ids]

nb_data = len(center)

left_indice = random.sample(range(nb_data), int(nb_data * LEFT_PERCENT))
right_indice = random.sample(range(nb_data), int(nb_data * RIGHT_PERCENT))

correction = 0.25

left_angles = []
left_images = []
for i in left_indice:
    left_images.append(left[i])
    left_angles.append(steering[i] + correction)

right_angles = []
right_images = []
for i in right_indice:
    right_images.append(left[i])
    right_angles.append(steering[i] - correction)

X_train = center + left_images + right_images
y_train = steering + left_angles + right_angles
# X_train = center
# y_train = steering
X_train, y_train = shuffle(X_train, y_train)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)


def plot():
    print("ploting")
    plt.title('original steering')
    plt.hist(steering, bins=100)
    plt.show()

    plt.title('Training data')
    plt.hist(y_train, bins=100)
    plt.show()


def plot_test():
    tmp_steerings = []
    for ephch in range(5):
        for i in range(len(X_train)):
            name = DATA_DIR + 'IMG/'+ X_train[i].split('/')[-1]
            original_image = cv2.imread(name)
            center_image = preprocess_image(original_image)
            center_angle = float(y_train[i])

            flip_coin = random.randint(0, 1)
            if flip_coin == 1:
                center_image, center_angle = flip(center_image, center_angle)
            trans = random.randint(0, 1)
            if trans:
                center_image, center_angle = translate(center_image, center_angle)
            tmp_steerings.append(center_angle)
    plt.title('test new data')
    plt.hist(tmp_steerings, bins=100)
    plt.show()


def flip(image, angle):
    new_image = cv2.flip(image, 1)
    new_angle = angle * -1
    return new_image, new_angle

TRANS_X_RANGE = 100
TRANS_Y_RANGE = 40
TRANS_ANGLE = 0.5


def translate(img, angle):
    """
    Shifts an image vertically and horizontally
    by a randaom amount.abs
    New angle is computed accordingly.
    """
    x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
    new_angle = angle + ((x_translation / TRANS_X_RANGE) * 2) * TRANS_ANGLE
    y_translation = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0])), new_angle


def preprocess_image(image):
    return cv2.resize(image[50:140, :], (80, 80))


def test_image(i=0):
    name = DATA_DIR + 'IMG/' + X_train[i].split('/')[-1]
    original_image = cv2.imread(name)
    plt.imshow(original_image)
    plt.savefig('orig.png')

    center_image = preprocess_image(original_image)
    center_angle = float(y_train[i])

    plt.imshow(center_image)
    plt.savefig('preprocess.png')

    flip_coin = random.randint(0, 1)
    if flip_coin == 1:
        center_image, center_angle = flip(center_image, center_angle)
    plt.imshow(center_image)
    plt.savefig('flip.png')
    plt.show()


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

                flip_coin = random.randint(0, 1)
                if flip_coin == 1:
                    center_image, center_angle = flip(center_image, center_angle)
                trans = random.randint(0, 1)
                if trans:
                    center_image, center_angle = translate(center_image, center_angle)
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            cur_X_train = np.array(images)
            cur_y_train = np.array(angles)
            yield shuffle(cur_X_train, cur_y_train)

# compile and train the model using the generator function
train_generator = generator(center, steering, batch_size=32)
validation_generator = generator(X_valid, y_valid, batch_size=32)

if __name__ == "__main__":
    # plot_test()
    plt.ion()
    # test_image()
    # exit(0)
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80, 80, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(20))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
    model.summary()

    model.fit_generator(train_generator, validation_data=validation_generator,
                        nb_epoch=FLAGS.epochs, samples_per_epoch=len(X_train),
                        nb_val_samples=len(X_valid))

    model.save('model.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
