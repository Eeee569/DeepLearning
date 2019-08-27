
import tensorflow as tf
tb  = tf.keras.callbacks.TensorBoard
import pickle
import time





pickle_in = open("../Data/X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("../Data/Y.pickle","rb")
Y = pickle.load(pickle_in)
pickle_in.close()
X = X/255.0


#simple logging using tensorboard
NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))
tensorboard = tb(log_dir='logs/{}'.format(NAME))



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = tf.keras.models.Sequential()
model.add(  tf.keras.layers.Conv2D(64,(3,3), input_shape = X.shape[1:]) )
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add(  tf.keras.layers.Conv2D(64,(3,3))  )
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())#3d featur maps to 1D feature vectors

model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation("relu"))

model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation("sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(X,Y,batch_size=50, epochs = 10, validation_split=0.1,callbacks=[tensorboard])
