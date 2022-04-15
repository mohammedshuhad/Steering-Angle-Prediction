import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, ReLU, ELU, LeakyReLU
from tensorflow.keras.initializers import Constant
import pandas as pd
from keras import backend as K
from matplotlib import pyplot as plt 

dim_x = 160
dim_y = 80
train_path = "/Users/shuhad/Downloads/archive_simulation/data/"
training_size = 8030
validation_size = 1000
epochs = 5
batch_size = 64
lr = 1.0e-4
model_name = "simulation_acc"
buffer_size = training_size - validation_size

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

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def activation_layer(ip, activation):
    return {'relu': ReLU()(ip),
            'elu': ELU()(ip),
            'lrelu': LeakyReLU()(ip)}[activation]
def conv2D(
    ip, filters, kernel_size, strides, layer_num, activation,
    kernel_initializer='he_uniform', bias_val=0.01
):

    conv_name = f'conv{layer_num}_{filters}_{kernel_size[0]}_{strides[0]}'
    bn_name = f'bn{layer_num}'

    layer = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_initializer=kernel_initializer,
                   bias_initializer=Constant(value=bias_val),
                   name=conv_name,)(ip)

    layer = BatchNormalization(name=bn_name)(layer)
    return activation_layer(ip=layer, activation=activation)
def fullyconnected_layers(
    ip, activation, inititalizer='he_uniform', bias_val=0.01
):

    layer = Dense(
        100, kernel_initializer=inititalizer,
        bias_initializer=Constant(value=bias_val), name='dense1'
    )(ip)

    layer = activation_layer(ip=layer, activation=activation)

    layer = Dense(
        50, kernel_initializer=inititalizer,
        bias_initializer=Constant(value=bias_val), name='dense2'
    )(layer)

    layer = activation_layer(ip=layer, activation=activation)

    return Dense(
        10, kernel_initializer=inititalizer,
        bias_initializer=Constant(value=bias_val), name='dense3'
    )(layer)

    return activation_layer(ip=layer, activation=activation)
def build_model(
    ip=tf.keras.Input(shape=(dim_y, dim_x, 3)), activation='relu', dropout=0.5,
    compile_model=True, lr=1e-3
):

    layer = conv2D(
        ip, filters=24, kernel_size=(5, 5), strides=(2, 2), layer_num=1,
        activation=activation
    )

    layer = conv2D(
        layer, filters=36, kernel_size=(5, 5), strides=(2, 2), layer_num=2,
        activation=activation
    )

    layer = conv2D(
        layer, filters=48, kernel_size=(5, 5), strides=(2, 2), layer_num=3,
        activation=activation
    )

    layer = conv2D(
        layer, filters=64, kernel_size=(3, 3), strides=(1, 1), layer_num=4,
        activation=activation
    )

    layer = conv2D(
        layer, filters=64, kernel_size=(3, 3), strides=(1, 1), layer_num=5,
        activation=activation
    )

    layer = Dropout(dropout)(layer)

    layer = Flatten()(layer)

    layer = fullyconnected_layers(layer, activation=activation)
    op_layer = Dense(1, name="op_layer")(layer)

    model = tf.keras.Model(ip, op_layer)
    if compile_model:
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=lr),run_eagerly = True)
    return model

x_train = []
y_train = []

df = pd.read_csv("/Users/shuhad/Downloads/archive_simulation/data/driving_log.csv")
x = df.loc[:,'center'].values
y = df.loc[:,'steering'].values


i = 0
while i < training_size:
    images = vectorized_imread(train_path + str(x[i]))
    images = vectorized_imresize(
        images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA
    )
    images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
    # images = images * (1./255)
    x_train.append(images)
    y_train.append(y[i])
    i += 1



x_val = x_train[-validation_size:]
y_val = y_train[-validation_size:]
x_train = x_train[:-validation_size]
y_train = y_train[:-validation_size]

x_train = np.array(x_train, np.float32)
x_valid = np.array(x_val, np.float32)
y_train = np.array(y_train, np.float32)
y_val = np.array(y_val, np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=validation_size).batch(batch_size)

print('Done Reading')


model = build_model(activation = 'elu', lr =lr)
model.summary()
result = model.fit(train_dataset,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=val_dataset,
          verbose=1)

model.save(model_name + '.h5')
model.save_weights(model_name + '_weights.h5')
# print('model saved')

# print(result)

# plt.plot(result.history['val_loss'])
# plt.plot(result.history['loss'])
# plt.title('model mse')
# plt.ylabel('mse')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # image = cv2.imread("1479425678173958759.jpg")
# # image = cv2.resize(image, (dim_x, dim_y))
# # image = image * (1./255)
# # predict_data = tf.data.Dataset.from_tensor_slices(image)
# # predict_data = predict_data.shuffle(buffer_size=1).batch(1)
# # print(model.predict(predict_data))