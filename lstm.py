import os 
import cv2
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Conv3D, MaxPooling3D, Dropout \
  ,BatchNormalization, ReLU, ELU, LeakyReLU, AveragePooling3D, Permute, LSTM, Reshape
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

vectorized_imread = np.vectorize(
    cv2.imread, signature="()->(x,y,z)"
)
vectorized_imresize = np.vectorize(
    cv2.resize, excluded=['dsize', 'interpolation'],
    signature='(x,y,z)->(a,b,c)'
)
vectorized_cvtColor = np.vectorize(
    cv2.cvtColor, excluded='code', signature="(x,y,z),()->(a,b,c)"
)
def channelwise_standardization(images, epsilon=1e-7):
    mean = np.mean(images, axis=(1, 2), keepdims=True)
    std = np.std(images, axis=(1, 2), keepdims=True)
    return (images - mean) / (std + epsilon)

x_train = []
y_train = []

df = pd.read_csv("/Users/shuhad/Downloads/archive_simulation/data/driving_log.csv")

dim_x = 160
dim_y = 80
dim_z = 4
dim_channel = 3
train_path = "/Users/shuhad/Downloads/archive_simulation/data/"
training_size = 8030
validation_size = 800
epochs = 5
batch_size = 64
lr = 1.0e-4
model_name = "../Models/lstm_full"

df = pd.read_csv(meta_path)
x = df.loc[:,'center'].values
r = df.loc[:, 'right'].values
l = df.loc[:, 'left'].values
y = df.loc[:,'steering'].values

i = dim_z

print('Started Reading')
while i < training_size:
    j = i - dim_z 
    x_frames = []
    while(j < i):
        images = vectorized_imread(train_path + str(x[j]))
        images = vectorized_imresize(
            images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA
        )
        images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
        x_frames.append(images)
        j += 1
    x_train.append(x_frames)
    y_train.append(y[i])
    i += 1
print("Reading Done")

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=training_size).batch(batch_size)

model = Sequential()
model.add(tf.keras.Input(shape = (dim_z, dim_y, dim_x, dim_channel)))

model.add(Conv3D(24, kernel_size = 5, strides = 2, input_shape = (dim_z, dim_y, dim_x, dim_channel), data_format = "channels_last", padding = "same"))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(36, kernel_size = 5, strides = 2, data_format = "channels_last", padding = "same"))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(48, kernel_size = 5, strides = 2, data_format = "channels_last", padding = "same"))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(64, kernel_size = 3, strides = 1, data_format = "channels_last", padding = "same"))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(64, kernel_size = 3, strides = 1, data_format = "channels_last", padding = "same"))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling3D(pool_size=(2,2,2), strides = 1, padding = "valid", data_format = None))

model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Reshape((1,10944)))
model.add(LSTM(64))

model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=lr),run_eagerly = True)
model.summary()

result = model.fit(train_dataset,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

model.save(model_name + '.h5')
model.save_weights(model_name + '_weights.h5')
print('model saved')

print(result)
, kernel_regularizer=tf.keras.regularizers.l2(0.0001)