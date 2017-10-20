# importing basic dependencies
import matplotlib.pyplot as plt # for seeing the images
import cv2 # for image processing
import glob # for file handling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output # to get the files in currect folder
from keras.utils import to_categorical # to convert to one-hot encodings
import tqdm # progress bar
from collections import Counter # for getting breed data

# Importing ML Dependencies
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

# now we load paths of the images
# loading images path --> train
images_train_path = '../input/train/*.jpg'
images_train_paths = glob.glob(images_train_path)
print(images_train_paths[0])

# laoding images path --> test
images_test_path = '../input/test/*.jpg'
images_test_paths = glob.glob(images_test_path)
print(images_test_paths[0])

# taking the labels for the images
labels = pd.read_csv('../input/labels.csv')
print(labels.head())

# taking the labels and converting to one hot
breeds = sorted(list(set(labels['breed'].values)))
# making a dictionary of breeds which will be used for one-hot encoding
b2id = dict((b,i) for i,b in enumerate(breeds))
# converting labeled breeds to numbers
breed_vector = [b2id[i] for i in labels['breed'].values]
# converting to one-hot encoding
data_y = to_categorical(breed_vector)

print('[*]Total images:', len(images_test_paths) + len(images_train_paths))
print('[*]Total training images:', len(images_train_paths))
print('[*]Total test images:', len(images_test_paths))
print('[*]Total breeds:',len(breeds))
print('[*]data_y.shape:', data_y.shape)
