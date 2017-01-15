import os
import numpy as np
import cv2

np.random.seed(1337)  # for reproducibility

MAX_ROW = 60
MAX_COL = 100


# the data, shuffled and split between train and test sets
def load_images_from_folder(folder, images, labels, label_value):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))  # From here I can read either in RGB or Grey scale.
        if img is not None:
            # res = resize_and_pad(img)
            res = cv2.resize(img, (MAX_COL, MAX_ROW), interpolation=cv2.INTER_CUBIC)
            images.append(res)
            labels.append(label_value)


def read_images(folder_positive, folder_negative):
    images = []
    labels = []
    load_images_from_folder(folder_positive, images, labels, 1)
    load_images_from_folder(folder_negative, images, labels, 0)
    data = np.asarray(images)
    y = np.asarray(labels)

    return data, y


def read_test_images(folder):
    images = []
    files = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))  # From here I can read either in RGB or Grey scale.
        if img is not None:
            # res = resize_and_pad(img)
            res = cv2.resize(img, (MAX_COL, MAX_ROW), interpolation=cv2.INTER_CUBIC)
            images.append(res)
            files.append(filename)
    images = np.asarray(images)
    return images, files
