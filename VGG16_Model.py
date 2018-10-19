
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import os
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import random as rn
import tensorflow as tf
import itertools

### Reproducibility ###

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(29)  # For numpy numbers
rn.seed(29)   # For Python
tf.set_random_seed(29)    #For Tensorflow



#Force tensorflow to use a single thread
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#Keras code goes after this



train_path = "/home/ubuntu/Plant-Species-Recognition/MLTutorial/cats_and_dogs/train/"
test_path = "/home/ubuntu/Plant-Species-Recognition/MLTutorial/cats_and_dogs/test/"
valid_path = "/home/ubuntu/Plant-Species-Recognition/MLTutorial/cats_and_dogs/valid/"

#classes = ['Charlock', 'Scentless Mayweed', 'Shepherds Purse', 'Loose Silky-bent', 'Common wheat', 'Maize', 'Black-grass', 'Small-flowered Cranesbill', 'Sugar beet', 'Cleavers', 'Common Chickweed', 'Fat Hen']

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224), classes=['dog', 'cat'] , batch_size = 10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224), classes=['dog', 'cat'], batch_size = 5)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size = (224,224), classes=['dog','cat'], batch_size = 5)


def plots(ims, figsize=(12, 6), rows=1, interp=False, title=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if title is not None:
            sp.set_title(title[i], fontsize=10)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

ims, labels = next(train_batches)
plots(ims, title = labels)

#### VGG 16 model ####
vgg16_model = keras.applications.vgg16.VGG16()
#
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation = 'softmax'))

model.summary()


#### Training VGG 16 Model ####

model.compile(Adam(lr = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])


model.fit_generator(train_batches, steps_per_epoch=35, validation_data = valid_batches,
                    validation_steps=20, epochs = 10, verbose = 2)

