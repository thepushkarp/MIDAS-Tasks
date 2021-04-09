import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic("matplotlib", " inline")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam


shutil.unpack_archive('trainPart1.zip', '../input/trainpart1zip')


image = Image.open('../input/trainpart1zip/train/Sample001/img001-001.png')
np_image = np.array(image)
print(np_image.shape)
print(image.mode)
imshow(image)


train_datagen1 = ImageDataGenerator(rescale=1./255, validation_split=0.2)


BATCH_SIZE = 64
IMAGE_SIZE = (45, 60)
EPOCHS = 400


train_generator1 = train_datagen1.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='training',
        seed=42,
        shuffle=True)


validation_generator1 = train_datagen1.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=tf.squeeze(IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='validation',
        seed=42,
        shuffle=True)


X_train_batch0, y_train_batch0 = train_generator1.next()
print(X_train_batch0.shape, y_train_batch0.shape)
print(y_train_batch0[0])
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_train_batch0[i]), cmap='gray')
plt.show() 


X_validation_batch0, y_validation_batch0 = validation_generator1.next()
print(X_validation_batch0.shape, y_validation_batch0.shape)
print(y_validation_batch0[0])
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_validation_batch0[i]), cmap='gray')
plt.show() 


model1 = Sequential()

# 1st Convolution Layer
model1.add(Conv2D(6, input_shape=(*IMAGE_SIZE, 1), kernel_size=(5,5), padding='same', activation='relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# 2nd Convolution Layer
model1.add(Conv2D(16, kernel_size=(5,5), activation='relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Passing to a Fully Connected Layer
model1.add(Flatten())

# 1st Fully Connected Layer
model1.add(Dense(256, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.4))

# 2nd Fully Connected Layer
model1.add(Dense(128, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.4))

# Output Layer
model1.add(Dense(62, activation='softmax'))


model1.summary()


model1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1)


checkpoint_filepath1 = 'exp1/checkpoint'
model_checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath1,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


history1 = model1.fit(
    train_generator1,
    epochs=EPOCHS,
    validation_data=validation_generator1,
    steps_per_epoch = train_generator1.samples // BATCH_SIZE,
    validation_steps = validation_generator1.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback1, early_stopping_callback]
)


plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


no_augmentation = ImageDataGenerator(rescale=1./255)

no_augmentation_gen = no_augmentation.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        seed=42,
        shuffle=True)

X_no_aug, _ = no_augmentation_gen.next()
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_no_aug[i]), cmap='gray')
plt.show() 


augmentation_test_rotation = ImageDataGenerator(rescale=1./255, rotation_range=15)

augmentation_test_rotation_gen = augmentation_test_rotation.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        seed=42,
        shuffle=True)

X_aug_rot, _ = augmentation_test_rotation_gen.next()
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_aug_rot[i]), cmap='gray')
plt.show() 


augmentation_test_shear = ImageDataGenerator(rescale=1./255, shear_range=0.3)

augmentation_test_shear_gen = augmentation_test_shear.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        seed=42,
        shuffle=True)

X_aug_shear, _ = augmentation_test_shear_gen.next()
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_aug_shear[i]), cmap='gray')
plt.show() 


augmentation_test_zoom = ImageDataGenerator(rescale=1./255, zoom_range=0.2)

augmentation_test_zoom_gen = augmentation_test_zoom.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        seed=42,
        shuffle=True)

X_aug_zoom, _ = augmentation_test_zoom_gen.next()
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_aug_zoom[i]), cmap='gray')
plt.show() 


augmentation_test_shift = ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range=0.2)

augmentation_test_shift_gen = augmentation_test_shift.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        seed=42,
        shuffle=True)

X_aug_shift, _ = augmentation_test_shift_gen.next()
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_aug_shift[i]), cmap='gray')
plt.show() 


train_datagen2 = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.1,
    rotation_range=0.5)

validation_datagen2 = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)


train_generator2 = train_datagen2.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='training',
        seed=42,
        shuffle=True)


validation_generator2 = validation_datagen2.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='validation',
        seed=42,
        shuffle=True)


X_train_batch0, y_train_batch0 = train_generator2.next()
print(X_train_batch0.shape, y_train_batch0.shape)
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_train_batch0[i]), cmap='gray')
plt.show() 


X_validation_batch0, y_validation_batch0 = validation_generator2.next()
print(X_validation_batch0.shape, y_validation_batch0.shape)
plt.figure(figsize=(16,12))
for i in range(1, 17):
    plt.subplot(4,4,i)
    imshow(tf.squeeze(X_validation_batch0[i]), cmap='gray')
plt.show() 


model2 = Sequential()

# 1st Convolution Layer
model2.add(Conv2D(6, input_shape=(*IMAGE_SIZE, 1), kernel_size=(5,5), padding='same', activation='relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2,2), strides=2))

# 2nd Convolution Layer
model2.add(Conv2D(16, kernel_size=(5,5), activation='relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Passing to a Fully Connected Layer
model2.add(Flatten())

# 1st Fully Connected Layer
model2.add(Dense(256, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.4))

# 2nd Fully Connected Layer
model2.add(Dense(128, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.4))

# Output Layer
model2.add(Dense(62, activation='softmax'))


model2.summary()


model2.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath2 = 'exp2/checkpoint'
model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath2,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


history2 = model2.fit(
    train_generator2,
    epochs=EPOCHS,
    validation_data=validation_generator2,
    steps_per_epoch = train_generator2.samples // BATCH_SIZE,
    validation_steps = validation_generator2.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback2, early_stopping_callback]
)


plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


# Mish Activation Function
def mish(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)


model3 = Sequential()

# 1st Convolution Layer
model3.add(Conv2D(6, input_shape=(*IMAGE_SIZE, 1),
                  kernel_size=(5,5), padding='same', activation=mish))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2,2), strides=2))

# 2nd Convolution Layer
model3.add(Conv2D(16, kernel_size=(5,5), activation=mish))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Passing to a Fully Connected Layer
model3.add(Flatten())

# 1st Fully Connected Layer
model3.add(Dense(256, activation=mish))
model3.add(BatchNormalization())
model3.add(Dropout(0.4))

# 2nd Fully Connected Layer
model3.add(Dense(128, activation=mish))
model3.add(BatchNormalization())
model3.add(Dropout(0.4))

# Output Layer
model3.add(Dense(62, activation='softmax'))


model3.summary()


model3.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath3 = 'exp3/checkpoint'
model_checkpoint_callback3 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath3,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


history3 = model3.fit(
    train_generator1,
    epochs=EPOCHS,
    validation_data=validation_generator1,
    steps_per_epoch = train_generator1.samples // BATCH_SIZE,
    validation_steps = validation_generator1.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback3, early_stopping_callback]
)


plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


model4 = Sequential()

# 1st Convolution Layer
model4.add(Conv2D(32, input_shape=(*IMAGE_SIZE, 1), kernel_size=3, activation=mish))
model4.add(BatchNormalization())
model4.add(Conv2D(32, kernel_size=3, activation=mish))
model4.add(BatchNormalization())
model4.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=mish))
model4.add(BatchNormalization())
model4.add(Dropout(0.4))

# 2nd Convolution Layer
model4.add(Conv2D(64, kernel_size=3, activation=mish))
model4.add(BatchNormalization())
model4.add(Conv2D(64, kernel_size=3, activation=mish))
model4.add(BatchNormalization())
model4.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=mish))
model4.add(BatchNormalization())
model4.add(Dropout(0.4))

# 3rd Convolution Layer
model4.add(Conv2D(128, kernel_size = 4, activation=mish))
model4.add(BatchNormalization())

# Passing to a Fully Connected Layer
model4.add(Flatten())
model4.add(Dropout(0.4))

# Output Layer
model4.add(Dense(62, activation='softmax'))


model4.summary()


model4.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath4 = 'exp4/checkpoint'
model_checkpoint_callback4 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath4,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


history4 = model4.fit(
    train_generator1,
    epochs=EPOCHS,
    validation_data=validation_generator1,
    steps_per_epoch = train_generator1.samples // BATCH_SIZE,
    validation_steps = validation_generator1.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback4, early_stopping_callback]
)


plt.plot(history4.history['accuracy'])
plt.plot(history4.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.plot(history4.history['loss'])
plt.plot(history4.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


checkpoint_filepath5 = 'exp5/checkpoint'
model_checkpoint_callback5 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath5,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


early_stopping_callback2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
    verbose=1)


from tensorflow.keras.applications import EfficientNetB0

# Initializer taken from the source code
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

# Input Layer
inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 1))

# Efficient layer except the top layer
x = EfficientNetB0(include_top=False, weights=None,
    input_shape=(*IMAGE_SIZE, 1))(inputs)

# Top

# Global Average Pooling Layer
x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Dropout(0.4, name='top_dropout')(x)

# Output Layer
outputs = tf.keras.layers.Dense(62,
    activation='softmax',
    kernel_initializer=DENSE_KERNEL_INITIALIZER,
    name='predictions')(x)

model5 = tf.keras.Model(inputs, outputs)

model5.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

model5.summary()


history5 = model5.fit(
    train_generator1,
    epochs=EPOCHS,
    validation_data=validation_generator1,
    steps_per_epoch = train_generator1.samples // BATCH_SIZE,
    validation_steps = validation_generator1.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback5, early_stopping_callback2]
)


plt.plot(history5.history['accuracy'])
plt.plot(history5.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.plot(history5.history['loss'])
plt.plot(history5.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


BATCH_SIZE = 32


train_generator3 = train_datagen1.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='training',
        seed=42,
        shuffle=True)

validation_generator3 = train_datagen1.flow_from_directory(
        '../input/trainpart1zip/train',
        target_size=tf.squeeze(IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='validation',
        seed=42,
        shuffle=True)


model6_1 = Sequential()

# 1st Convolution Layer
model6_1.add(Conv2D(6, input_shape=(*IMAGE_SIZE, 1),
                  kernel_size=(5,5), padding='same', activation=mish))
model6_1.add(BatchNormalization())
model6_1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# 2nd Convolution Layer
model6_1.add(Conv2D(16, kernel_size=(5,5), activation=mish))
model6_1.add(BatchNormalization())
model6_1.add(MaxPooling2D(pool_size=(2,2), strides=2))

# Passing to a Fully Connected Layer
model6_1.add(Flatten())

# 1st Fully Connected Layer
model6_1.add(Dense(256, activation=mish))
model6_1.add(BatchNormalization())
model6_1.add(Dropout(0.4))

# 2nd Fully Connected Layer
model6_1.add(Dense(128, activation=mish))
model6_1.add(BatchNormalization())
model6_1.add(Dropout(0.4))

# Output Layer
model6_1.add(Dense(62, activation='softmax'))


model6_1.summary()


model6_1.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath6_1 = 'exp6_1/checkpoint'
model_checkpoint_callback6_1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath6_1,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


history6_1 = model6_1.fit(
    train_generator3,
    epochs=EPOCHS,
    validation_data=validation_generator3,
    steps_per_epoch = train_generator3.samples // BATCH_SIZE,
    validation_steps = validation_generator3.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback6_1, early_stopping_callback]
)


plt.plot(history6_1.history['accuracy'])
plt.plot(history6_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.plot(history6_1.history['loss'])
plt.plot(history6_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


model6_2 = Sequential()

# 1st Convolution Layer
model6_2.add(Conv2D(32, input_shape=(*IMAGE_SIZE, 1), kernel_size=3, activation=mish))
model6_2.add(BatchNormalization())
model6_2.add(Conv2D(32, kernel_size=3, activation=mish))
model6_2.add(BatchNormalization())
model6_2.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=mish))
model6_2.add(BatchNormalization())
model6_2.add(Dropout(0.4))

# 2nd Convolution Layer
model6_2.add(Conv2D(64, kernel_size=3, activation=mish))
model6_2.add(BatchNormalization())
model6_2.add(Conv2D(64, kernel_size=3, activation=mish))
model6_2.add(BatchNormalization())
model6_2.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=mish))
model6_2.add(BatchNormalization())
model6_2.add(Dropout(0.4))

# 3rd Convolution Layer
model6_2.add(Conv2D(128, kernel_size = 4, activation=mish))
model6_2.add(BatchNormalization())

# Passing to a Fully Connected Layer
model6_2.add(Flatten())
model6_2.add(Dropout(0.4))

# Output Layer
model6_2.add(Dense(62, activation='softmax'))


model6_2.summary()


model6_2.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


checkpoint_filepath6_2 = 'exp6_2/checkpoint'
model_checkpoint_callback6_2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath6_2,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


history6_2 = model6_2.fit(
    train_generator3,
    epochs=EPOCHS,
    validation_data=validation_generator3,
    steps_per_epoch = train_generator3.samples // BATCH_SIZE,
    validation_steps = validation_generator3.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback6_2, early_stopping_callback]
)


plt.plot(history6_2.history['accuracy'])
plt.plot(history6_2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


plt.plot(history6_2.history['loss'])
plt.plot(history6_2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()



