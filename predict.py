import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, ReLU, ELU, LeakyReLU
from tensorflow.keras.initializers import Constant
import pandas as pd
from keras import backend as K

trained_model_path = 'Model_H.h5'
test_path = "../Udacity/center/"

model = load_model(trained_model_path)

model.summary()

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

df = pd.read_csv("../Udacity/final_example.csv")
x = df.loc[:,'frame_id'].values
y = df.loc[:,'steering_angle'].values

test_size = 10

dim_x = 160
dim_y = 120
x_test = []
y_actual = []

i= 0
while(i < test_size):
    images = vectorized_imread(test_path + str(x[i]) + '.jpg')
    images = vectorized_imresize(
        images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA
    )
    images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
    x_test.append(images)
    y_actual.append(y[i])
    i += 1


x_test = np.array(x_test, np.float32)
y_actual = np.array(y_actual, np.float32)
predict_data = tf.data.Dataset.from_tensor_slices(x_test)
predict_data = predict_data.batch(64)
result = model.predict(predict_data)
error = result - y_actual
print(result)