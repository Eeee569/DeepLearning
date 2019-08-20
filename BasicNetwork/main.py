import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_trian, y_trian),(x_test, y_test)  = mnist.load_data()

x_trian = tf.keras.utils.normalize(x_trian, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


#building and runnint thenetwork
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_trian,y_trian, epochs=3)

#evaluating the the network
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


#saving the network and loading it again
model.save('basic_model.model')
new_model = tf.keras.models.load_model('basic_model.model')

predictions = new_model.predict(x_test)

#get the prediction for the first item in x_test ie x_test[0]
print(np.argmax(predictions[0]))