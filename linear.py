import tensorflow as tf
import numpy as np
from tensorflow import keras
model0 = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model0.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model0.fit(xs, ys, epochs=250)


model1 = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model1.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1, 2, 3, 4, 5, 10], dtype=float)
ys = np.array([100, 150, 200, 250, 300, 550], dtype=float)
model1.fit(xs, ys, epochs=200)
print()


print(model0.predict([10.0])[0][0], model1.predict([7.0])[0][0])
