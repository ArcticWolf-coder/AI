import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200)
print(tf.__version__)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.9):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True


callback = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
plt.imshow(training_images[0])
training_images = training_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(
                                        128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, callbacks=[callback])
model.evaluate(test_images, test_labels)

print(training_labels[0])
# plt.show()
