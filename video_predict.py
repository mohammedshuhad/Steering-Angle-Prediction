from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import argparse
import numpy as np
import time
import cv2
import tensorflow as tf
from operator import itemgetter
import pandas as pd
import math

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

dim_x = 160
dim_y = 80


df = pd.read_csv("/Users/shuhad/Downloads/archive_simulation/data/driving_log.csv")
x = df.loc[:,'center'].values
y = df.loc[:,'steering'].values
test_size = 200
model = load_model("Model_H.h5")
train_path = "/Users/shuhad/Downloads/archive_simulation/data/"
i= 0

while(i < test_size):
    images = cv2.imread(train_path + str(x[i]))
    orig = cv2.imread(train_path + str(x[i]))
    images = vectorized_imresize(
        images, dsize=(dim_x, dim_y), interpolation=cv2.INTER_AREA
    )
    images = vectorized_cvtColor(images, cv2.COLOR_BGR2YUV)
    images = np.expand_dims(images, axis=0)
    x_frames = np.array(x_frames, np.float32)
    x_frames = np.expand_dims(x_frames, axis=0)
    print(x_frames.shape)
    angle = model.predict(x_frames)
    label = "{} {}".format(angle, y[i], math.sqrt((y[i] - angle)*((y[i] - angle))))
    orig = cv2.putText(orig, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)	
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    i += 1