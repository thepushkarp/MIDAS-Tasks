import shutil
from PIL import Image
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic("matplotlib", " inline")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam


shutil.unpack_archive('trainPart1.zip', 'input/part2')


for i in range(11, 63):
    shutil.rmtree(f'input/part2/train/Sample0{i}')


BATCH_SIZE = 64
IMAGE_SIZE_BEFORE_PADDING = (21, 28)
EPOCHS = 400
IMAGE_SIZE = (28, 28)


def pad_top_nd_bottom(image):
    '''
    Pads the top and bottom of the image with zeros
    
    Arguments:
        image: Image of dimension 21x28x1
    
    Returns:
        padded_image: Image of dimension 28x28x1 with zero padding added at top and bottom
    '''
    padded_image = tf.image.resize_with_crop_or_pad(
        image, 28, 28)
    return padded_image


train_datagen_ours = ImageDataGenerator(rescale=1./255)


train_generator_ours = train_datagen_ours.flow_from_directory(
    'input/part2/train',
    target_size=IMAGE_SIZE_BEFORE_PADDING,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training',
    seed=42,
    shuffle=True)

X_train_batch0, y_train_batch0 = train_generator_ours.next()
print(X_train_batch0.shape, y_train_batch0.shape)
print(y_train_batch0[0])
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_train_batch0[i]), cmap='gray')
plt.show()


# Softmax Temperature
temp = 5


model1_pretrained = Sequential()

# Lambda Layer for adding Padding
model1_pretrained.add(Lambda(lambda image: tf.image.resize_with_crop_or_pad(
        image, 28, 28)))

# 1st Convolution Layer
model1_pretrained.add(Conv2D(6, input_shape=(*IMAGE_SIZE, 1),
kernel_size=(5,5), padding='same', activation=mish))
model1_pretrained.add(BatchNormalization())
model1_pretrained.add(MaxPooling2D(pool_size=(2,2), strides=2))

# 2nd Convolution Layer
model1_pretrained.add(Conv2D(16, kernel_size=(5,5), activation=mish))
model1_pretrained.add(BatchNormalization())
model1_pretrained.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Passing to a Fully Connected Layer
model1_pretrained.add(Flatten())

# 1st Fully Connected Layer
model1_pretrained.add(Dense(256, activation=mish))
model1_pretrained.add(BatchNormalization())
model1_pretrained.add(Dropout(0.4))

# 2nd Fully Connected Layer
model7.add(Dense(128, activation=mish))
model7.add(BatchNormalization())
model7.add(Dropout(0.4))

# Output Layer

# Increasing the softmax temperature
model7.add(Lambda(lambda x: x / temp))
model7.add(Dense(62, activation='softmax'))
