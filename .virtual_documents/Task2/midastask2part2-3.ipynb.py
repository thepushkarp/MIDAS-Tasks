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
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


shutil.unpack_archive('trainPart1.zip', 'input/part2')


for i in range(11, 63):
    shutil.rmtree(f'input/part2/train/Sample0{i}')


BATCH_SIZE = 64
IMAGE_SIZE_BEFORE_PADDING = (21, 28)
EPOCHS = 400
IMAGE_SIZE = (28, 28)


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, preprocessing_function=lambda x: 1-x)


train_generator = train_datagen.flow_from_directory(
    'input/part2/train',
    target_size=IMAGE_SIZE_BEFORE_PADDING,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training',
    seed=42,
    shuffle=True)

validation_generator = train_datagen.flow_from_directory(
    'input/part2/train',
    target_size=IMAGE_SIZE_BEFORE_PADDING,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation',
    seed=42,
    shuffle=True)

X_train_batch0, y_train_batch0 = train_generator.next()
print(X_train_batch0.shape, y_train_batch0.shape)
print(y_train_batch0[0])
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_train_batch0[i]), cmap='gray')
plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
imshow(tf.squeeze(x_train[0]), cmap='gray')


mnist_datagen = ImageDataGenerator(rescale=1.0/255.0)

# prepare an iterators to scale images
mnist_train_gen = mnist_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
mnist_test_gen = mnist_datagen.flow(x_test, y_test, batch_size=BATCH_SIZE)

mnist_x_train, _ = mnist_train_gen.next()
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(mnist_x_train[i]), cmap='gray')
plt.show()


# Early Stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
    verbose=1)

early_stopping_callback2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
    verbose=1)


# Softmax Temperature
temp = 5


# Mish Activation function
def mish(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)


model1 = Sequential()

# Lambda Layer for adding Padding
model1.add(Lambda(lambda image: tf.image.resize_with_crop_or_pad(
        image, 28, 28), input_shape=(*IMAGE_SIZE_BEFORE_PADDING, 1)))

# 1st Convolution Layer
model1.add(Conv2D(6, input_shape=(*IMAGE_SIZE, 1),
                  kernel_size=(5,5), padding='same', activation=mish))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# 2nd Convolution Layer
model1.add(Conv2D(16, kernel_size=(5,5), activation=mish))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Passing to a Fully Connected Layer
model1.add(Flatten())

# 1st Fully Connected Layer
model1.add(Dense(120, activation=mish))
model1.add(BatchNormalization())
model1.add(Dropout(0.4))

# 2nd Fully Connected Layer
model1.add(Dense(84, activation=mish))
model1.add(BatchNormalization())
model1.add(Dropout(0.4))

# Output Layer
# Increasing the softmax temperature
model1.add(Lambda(lambda x: x / temp))
model1.add(Dense(10, activation='softmax'))

model1.summary()

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


checkpoint_filepath1 = 'part2_pretrained/checkpoint'
model_checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath1,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history1 = model1.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_steps = validation_generator.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback1, early_stopping_callback]
)


plt.figure(1)
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


model1 = Sequential()

# Lambda Layer for adding Padding
model1.add(Lambda(lambda image: tf.image.resize_with_crop_or_pad(
        image, 28, 28), input_shape=(*IMAGE_SIZE_BEFORE_PADDING, 1)))

# 1st Convolution Layer
model1.add(Conv2D(6, input_shape=(*IMAGE_SIZE, 1),
                  kernel_size=(5,5), padding='same', activation=mish))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# 2nd Convolution Layer
model1.add(Conv2D(16, kernel_size=(5,5), activation=mish))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Passing to a Fully Connected Layer
model1.add(Flatten())

# 1st Fully Connected Layer
model1.add(Dense(120, activation=mish))
model1.add(BatchNormalization())
model1.add(Dropout(0.4))

# 2nd Fully Connected Layer
model1.add(Dense(84, activation=mish))
model1.add(BatchNormalization())
model1.add(Dropout(0.4))

# Output Layer
# Increasing the softmax temperature
model1.add(Lambda(lambda x: x / temp))
model1.add(Dense(10, activation='softmax'))

model1.summary()

model1.load_weights('part2_pretrained/checkpoint')

model1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath1_after = 'part2_after_training/checkpoint'
model_checkpoint_callback1_after = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath1_after,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history1_after = model1.fit(
    mnist_train_gen,
    epochs=EPOCHS,
    validation_data=mnist_test_gen,
    steps_per_epoch = len(x_train) // BATCH_SIZE,
    validation_steps = len(x_test) // BATCH_SIZE,
    callbacks=[model_checkpoint_callback1_after, early_stopping_callback]
)


plt.figure(1)
plt.plot(history1_after.history['accuracy'])
plt.plot(history1_after.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history1_after.history['loss'])
plt.plot(history1_after.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


model1 = Sequential()

# Lambda Layer for adding Padding
model1.add(Lambda(lambda image: tf.image.resize_with_crop_or_pad(
        image, 28, 28), input_shape=(*IMAGE_SIZE_BEFORE_PADDING, 1)))

# 1st Convolution Layer
model1.add(Conv2D(6, input_shape=(*IMAGE_SIZE, 1),
                  kernel_size=(5,5), padding='same', activation=mish))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# 2nd Convolution Layer
model1.add(Conv2D(16, kernel_size=(5,5), activation=mish))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Passing to a Fully Connected Layer
model1.add(Flatten())

# 1st Fully Connected Layer
model1.add(Dense(120, activation=mish))
model1.add(BatchNormalization())
model1.add(Dropout(0.4))

# 2nd Fully Connected Layer
model1.add(Dense(84, activation=mish))
model1.add(BatchNormalization())
model1.add(Dropout(0.4))

# Output Layer
# Increasing the softmax temperature
model1.add(Lambda(lambda x: x / temp))
model1.add(Dense(10, activation='softmax'))

model1.summary()

model1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath1_un = 'part2_untrained/checkpoint'
model_checkpoint_callback1_un = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath1_un,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history1_un = model1.fit(
    mnist_train_gen,
    epochs=EPOCHS,
    validation_data=mnist_test_gen,
    steps_per_epoch = len(x_train) // BATCH_SIZE,
    validation_steps = len(x_test) // BATCH_SIZE,
    callbacks=[model_checkpoint_callback1_un, early_stopping_callback]
)


plt.figure(1)
plt.plot(history1_un.history['accuracy'])
plt.plot(history1_un.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history1_un.history['loss'])
plt.plot(history1_un.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.figure(1)
plt.plot(history1_after.history['accuracy'])
plt.plot(history1_un.history['accuracy'])
plt.title('training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['pretrained_training', 'untrained_training'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history1_after.history['val_accuracy'])
plt.plot(history1_un.history['val_accuracy'])
plt.title('validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['pretrained_validation', 'untrained_validation'], loc='best')
plt.show()

plt.figure(3)
plt.plot(history1_after.history['loss'])
plt.plot(history1_un.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['pretrained_validation', 'untrained_validation'], loc='best')
plt.show()

plt.figure(4)
plt.plot(history1_after.history['val_loss'])
plt.plot(history1_un.history['val_loss'])
plt.title('validation accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['pretrained_validation', 'untrained_validation'], loc='best')
plt.show()


model2 = Sequential()

# Lambda Layer for adding Padding
model2.add(Lambda(lambda image: tf.image.resize_with_crop_or_pad(
        image, 28, 28), input_shape=(*IMAGE_SIZE_BEFORE_PADDING, 1)))

# 1st Convolution Layer
model2.add(Conv2D(32, input_shape=(*IMAGE_SIZE, 1), kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=mish))
model2.add(BatchNormalization())
model2.add(Dropout(0.4))

# 2nd Convolution Layer
model2.add(Conv2D(64, kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(64, kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=mish))
model2.add(BatchNormalization())
model2.add(Dropout(0.4))

# 3rd Convolution Layer
model2.add(Conv2D(128, kernel_size = 4, activation=mish))
model2.add(BatchNormalization())

# Passing to a Fully Connected Layer
model2.add(Flatten())
model2.add(Dropout(0.4))

# Output Layer

# Increasing the softmax temperature
model2.add(Lambda(lambda x: x / temp))
model2.add(Dense(10, activation='softmax'))

model2.summary()

model2.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath2 = 'part2_pretrained2/checkpoint'
model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath2,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history2 = model2.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_steps = validation_generator.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback2, early_stopping_callback2]
)


plt.figure(1)
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


model2 = Sequential()

# Lambda Layer for adding Padding
model2.add(Lambda(lambda image: tf.image.resize_with_crop_or_pad(
        image, 28, 28), input_shape=(*IMAGE_SIZE_BEFORE_PADDING, 1)))

# 1st Convolution Layer
model2.add(Conv2D(32, input_shape=(*IMAGE_SIZE, 1), kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=mish))
model2.add(BatchNormalization())
model2.add(Dropout(0.4))

# 2nd Convolution Layer
model2.add(Conv2D(64, kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(64, kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=mish))
model2.add(BatchNormalization())
model2.add(Dropout(0.4))

# 3rd Convolution Layer
model2.add(Conv2D(128, kernel_size = 4, activation=mish))
model2.add(BatchNormalization())

# Passing to a Fully Connected Layer
model2.add(Flatten())
model2.add(Dropout(0.4))

# Output Layer

# Increasing the softmax temperature
model2.add(Lambda(lambda x: x / temp))
model2.add(Dense(10, activation='softmax'))

model2.summary()

model2.load_weights('part2_pretrained2/checkpoint')

model2.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath2_after = 'part2_after_training2/checkpoint'
model_checkpoint_callback2_after = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath2_after,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history2_after = model2.fit(
    mnist_train_gen,
    epochs=EPOCHS,
    validation_data=mnist_test_gen,
    steps_per_epoch = len(x_train) // BATCH_SIZE,
    validation_steps = len(x_test) // BATCH_SIZE,
    callbacks=[model_checkpoint_callback2_after, early_stopping_callback]
)


plt.figure(1)
plt.plot(history2_after.history['accuracy'])
plt.plot(history2_after.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history2_after.history['loss'])
plt.plot(history2_after.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


model2 = Sequential()

# Lambda Layer for adding Padding
model2.add(Lambda(lambda image: tf.image.resize_with_crop_or_pad(
        image, 28, 28), input_shape=(*IMAGE_SIZE_BEFORE_PADDING, 1)))

# 1st Convolution Layer
model2.add(Conv2D(32, input_shape=(*IMAGE_SIZE, 1), kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=mish))
model2.add(BatchNormalization())
model2.add(Dropout(0.4))

# 2nd Convolution Layer
model2.add(Conv2D(64, kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(64, kernel_size=3, activation=mish))
model2.add(BatchNormalization())
model2.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=mish))
model2.add(BatchNormalization())
model2.add(Dropout(0.4))

# 3rd Convolution Layer
model2.add(Conv2D(128, kernel_size = 4, activation=mish))
model2.add(BatchNormalization())

# Passing to a Fully Connected Layer
model2.add(Flatten())
model2.add(Dropout(0.4))

# Output Layer

# Increasing the softmax temperature
model2.add(Lambda(lambda x: x / temp))
model2.add(Dense(10, activation='softmax'))

model2.summary()

model2.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath2_un = 'part2_untrained2/checkpoint'
model_checkpoint_callback2_un = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath2_un,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history2_un = model2.fit(
    train_generator3,
    epochs=17,
    validation_data=mnist_test_gen,
    steps_per_epoch = len(x_train) // BATCH_SIZE,
    validation_steps = len(x_test) // BATCH_SIZE,
    callbacks=[model_checkpoint_callback2_un, early_stopping_callback]
)


plt.figure(1)
plt.plot(history2_un.history['accuracy'])
plt.plot(history2_un.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history2_un.history['loss'])
plt.plot(history2_un.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.figure(1)
plt.plot(history2_after.history['accuracy'])
plt.plot(history2_un.history['accuracy'])
plt.title('training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['pretrained_training', 'untrained_training'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history2_after.history['val_accuracy'])
plt.plot(history2_un.history['val_accuracy'])
plt.title('validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['pretrained_validation', 'untrained_validation'], loc='best')
plt.show()

plt.figure(3)
plt.plot(history2_after.history['loss'])
plt.plot(history2_un.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['pretrained_validation', 'untrained_validation'], loc='best')
plt.show()

plt.figure(4)
plt.plot(history2_after.history['val_loss'])
plt.plot(history2_un.history['val_loss'])
plt.title('validation accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['pretrained_validation', 'untrained_validation'], loc='best')
plt.show()


shutil.unpack_archive('mnistTask3.zip', 'input/part3')


train_datagen3 = ImageDataGenerator(rescale=1./255)


train_generator3 = train_datagen3.flow_from_directory(
    'input/part3/mnistTask',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training',
    seed=42,
    shuffle=True)

X_train_batch3, y_train_batch3 = train_generator3.next()
print(X_train_batch3.shape, y_train_batch3.shape)
print(y_train_batch3[0])
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_train_batch3[i]), cmap='gray')
plt.show()


model3 = Sequential()

# 1st Convolution Layer
model3.add(Conv2D(32, input_shape=(*IMAGE_SIZE, 1), kernel_size=3, activation=mish))
model3.add(BatchNormalization())
model3.add(Conv2D(32, kernel_size=3, activation=mish))
model3.add(BatchNormalization())
model3.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=mish))
model3.add(BatchNormalization())
model3.add(Dropout(0.4))

# 2nd Convolution Layer
model3.add(Conv2D(64, kernel_size=3, activation=mish))
model3.add(BatchNormalization())
model3.add(Conv2D(64, kernel_size=3, activation=mish))
model3.add(BatchNormalization())
model3.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=mish))
model3.add(BatchNormalization())
model3.add(Dropout(0.4))

# 3rd Convolution Layer
model3.add(Conv2D(128, kernel_size = 4, activation=mish))
model3.add(BatchNormalization())

# Passing to a Fully Connected Layer
model3.add(Flatten())
model3.add(Dropout(0.4))

# Output Layer

# Increasing the softmax temperature
model3.add(Lambda(lambda x: x / temp))
model3.add(Dense(10, activation='softmax'))

model3.summary()

model3.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath3 = 'part3/checkpoint'
model_checkpoint_callback3 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath3,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history3 = model3.fit(
    train_generator3,
    epochs=EPOCHS,
    validation_data=mnist_test_gen,
    steps_per_epoch = train_generator3.samples // BATCH_SIZE // BATCH_SIZE,
    validation_steps = len(x_test) // BATCH_SIZE,
    callbacks=[model_checkpoint_callback3, early_stopping_callback2]
)


plt.figure(1)
plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


model3 = Sequential()

# Lambda Layer for adding Padding
model3.add(Lambda(lambda image: tf.image.resize_with_crop_or_pad(
        image, 28, 28), input_shape=(*IMAGE_SIZE, 1)))

# 1st Convolution Layer
model3.add(Conv2D(32, input_shape=(*IMAGE_SIZE, 1), kernel_size=3, activation=mish))
model3.add(BatchNormalization())
model3.add(Conv2D(32, kernel_size=3, activation=mish))
model3.add(BatchNormalization())
model3.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=mish))
model3.add(BatchNormalization())
model3.add(Dropout(0.4))

# 2nd Convolution Layer
model3.add(Conv2D(64, kernel_size=3, activation=mish))
model3.add(BatchNormalization())
model3.add(Conv2D(64, kernel_size=3, activation=mish))
model3.add(BatchNormalization())
model3.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=mish))
model3.add(BatchNormalization())
model3.add(Dropout(0.4))

# 3rd Convolution Layer
model3.add(Conv2D(128, kernel_size = 4, activation=mish))
model3.add(BatchNormalization())

# Passing to a Fully Connected Layer
model3.add(Flatten())
model3.add(Dropout(0.4))

# Output Layer

# Increasing the softmax temperature
model3.add(Lambda(lambda x: x / temp))
model3.add(Dense(10, activation='softmax'))

model3.summary()

model3.load_weights('part2_pretrained2/checkpoint')

model3.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath3un = 'part3_un/checkpoint'
model_checkpoint_callback3un = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath3un,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history3un = model3.fit(
    train_generator3,
    epochs=EPOCHS,
    validation_data=mnist_test_gen,
    steps_per_epoch = train_generator3.samples // BATCH_SIZE // BATCH_SIZE,
    validation_steps = len(x_test) // BATCH_SIZE,
    callbacks=[model_checkpoint_callback3un, early_stopping_callback2]
)


plt.figure(1)
plt.plot(history3un.history['accuracy'])
plt.plot(history3un.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history3un.history['loss'])
plt.plot(history3un.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.figure(1)
plt.plot(history3.history['accuracy'])
plt.plot(history3un.history['accuracy'])
plt.title('training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['pretrained_training', 'untrained_training'], loc='best')
plt.show()

plt.figure(2)
plt.plot(history3.history['val_accuracy'])
plt.plot(history3un.history['val_accuracy'])
plt.title('validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['pretrained_validation', 'untrained_validation'], loc='best')
plt.show()

plt.figure(3)
plt.plot(history3.history['loss'])
plt.plot(history3un.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['pretrained_validation', 'untrained_validation'], loc='best')
plt.show()

plt.figure(4)
plt.plot(history3.history['val_loss'])
plt.plot(history3un.history['val_loss'])
plt.title('validation accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['pretrained_validation', 'untrained_validation'], loc='best')
plt.show()



