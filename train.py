from readline import append_history_file
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, ReLU, ELU, LeakyReLU
from tensorflow.keras.initializers import Constant
import pandas as pd
from keras import backend as K
from matplotlib import pyplot as plt 
from timeit import default_timer as timer
from tensorflow.keras.applications.resnet50 import ResNet50

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

def flip_images(images, labels, mask=None, threshold=0.5):
    to_be_flipped = (np.random.rand(len(images)) < threshold)
    mask = to_be_flipped if mask is None else mask & to_be_flipped
    images[mask] = np.flipud(images[mask])
    labels[mask] = -labels[mask]
    return images, labels

data_type = "simulation"

model_dir = "../Models/"
if(data_type == "simulation"):
    train_path = "/Users/shuhad/Downloads/archive_simulation/data/"
    meta_path = train_path + "driving_log.csv"
elif (data_type == "real"):
    train_path = "/Users/shuhad/Downloads/realdriving_data/"
    meta_path = train_path + "data.txt"

dim_x = 160
dim_y = 80
training_size = 8000
validation_size = 800 
epochs = 10
batch_size = 64
lr = 1.0e-4

model_name = "Model_H"
buffer_size = training_size - validation_size

x_train = []
y_train = []
x_valid = []
y_valid = []

if(data_type == "simulation"):
    df = pd.read_csv(meta_path)
    x = df.loc[:,'center'].values
    r = df.loc[:, 'right'].values
    l = df.loc[:, 'left'].values
    y = df.loc[:,'steering'].values
elif (data_type == "real"):
    df = pd.read_csv(meta_path, sep=" ", header=None)
    df.columns = ["a", "b"]
    x = df.a.values
    y = df.b.values

y = y*25
shift = 0.2 * 25

start_t = timer()

train_indexes = np.arange(training_size)
val_indexes = train_indexes[-validation_size:]
for k in val_indexes:
  train_indexes = np.delete(train_indexes, np.where(train_indexes == k))

for i in train_indexes:
    images = vectorized_imread(train_path + str(x[i]))
    # images = np.flipud(images)
    images = vectorized_imresize(images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA)
    images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
    images = channelwise_standardization(images, epsilon=1e-7)
    x_train.append(images)

    images = vectorized_imread(train_path + str(r[i]).lstrip(' '))
    # images = np.flipud(images)
    images = vectorized_imresize(images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA)
    images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
    images = channelwise_standardization(images, epsilon=1e-7)
    x_train.append(images)

    images = vectorized_imread(train_path + str(l[i]).lstrip(' '))
    # images = np.flipud(images)
    images = vectorized_imresize(images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA)
    images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
    images = channelwise_standardization(images, epsilon=1e-7)
    x_train.append(images)
     
    y[i] = y[i] * -1 

    y_train.append(y[i])
    y_train.append(y[i] - shift)
    y_train.append(y[i] + shift)

for j in val_indexes:
  images = vectorized_imread(train_path + str(x[j]))
#   images = np.flipud(images)
  images = vectorized_imresize(images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA)
  images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
  images = channelwise_standardization(images, epsilon=1e-7)
  x_valid.append(images)

  images = vectorized_imread(train_path + str(r[j]).lstrip(' '))
#   images = np.flipud(images)
  images = vectorized_imresize(images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA)
  images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
  images = channelwise_standardization(images, epsilon=1e-7)
  x_valid.append(images)

  images = vectorized_imread(train_path + str(l[j]).lstrip(' '))
#   images = np.flipud(images)
  images = vectorized_imresize(images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA)
  images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
  images = channelwise_standardization(images, epsilon=1e-7)
  x_valid.append(images)

  y[j] = y[j] * -1 

  y_valid.append(y[j])
  y_valid.append(y[j] - shift)
  y_valid.append(y[j] + shift)



x_train = np.array(x_train, np.float32)
x_valid = np.array(x_valid, np.float32)
y_train = np.array(y_train, np.float32)
y_valid = np.array(y_valid, np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
val_dataset = val_dataset.batch(batch_size)

base_model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(dim_y, dim_x, 3),include_top=False) 
base_model.trainable = False

activation_fn = "relu"

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model = Sequential()

# inputs = tf.keras.Input(shape=(dim_y, dim_x, 3))
# layers = base_model(inputs, training=False)
# model.add(base_model)
# model.add(Dense(1000))
# model.add(Activation(activation_fn))
# model.add(Dense(512))
# model.add(Activation(activation_fn))
# model.add(Dense(256))
# model.add(Activation(activation_fn))
# model.add(Dense(64))
# model.add(Activation(activation_fn))
# model.add(Dense(1))     

model.add(tf.keras.Input(shape = (dim_y, dim_x, 3)))
model.add(BatchNormalization())
model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2)), kernel_regularizer=tf.keras.regularizers.l2(0.0001))
model.add(Activation(activation_fn))
model.add(BatchNormalization())
model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2)), kernel_regularizer=tf.keras.regularizers.l2(0.0001))
model.add(Activation(activation_fn))
model.add(BatchNormalization())
model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2)), kernel_regularizer=tf.keras.regularizers.l2(0.0001))
model.add(Activation(activation_fn))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1)), kernel_regularizer=tf.keras.regularizers.l2(0.0001))
model.add(Activation(activation_fn))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1)), kernel_regularizer=tf.keras.regularizers.l2(0.0001))
model.add(Activation(activation_fn))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1000))
model.add(Activation(activation_fn))
model.add(Dense(100))
model.add(Activation(activation_fn))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=lr))
model.summary()
result = model.fit(train_dataset,batch_size=batch_size,epochs=epochs,validation_data=val_dataset,callbacks=[callback],verbose=1)

model.save(model_dir + model_name + '.h5')
model.save_weights(model_dir + model_name + '_weights.h5')

print('model saved')

print(result)

end_t = timer()
print("elapsed time = {} seconds".format(end_t-start_t))

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()