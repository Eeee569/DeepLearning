
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




dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))

            #simple logging using tensorboard
            tensorboard = tb(log_dir='logs/{}'.format(NAME))




            model = tf.keras.models.Sequential()
            model.add(  tf.keras.layers.Conv2D(layer_size,(3,3), input_shape = X.shape[1:]) )
            model.add(tf.keras.layers.Activation("relu"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))


            for l  in range(conv_layer-1):

                model.add(  tf.keras.layers.Conv2D(layer_size,(3,3))  )
                model.add(tf.keras.layers.Activation("relu"))
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))


            model.add(tf.keras.layers.Flatten())#3d featur maps to 1D feature vectors

            for l in range(dense_layer):

                model.add(tf.keras.layers.Dense(layer_size))
                model.add(tf.keras.layers.Activation("relu"))


            model.add(tf.keras.layers.Dense(1))
            model.add(tf.keras.layers.Activation("sigmoid"))

            model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
            model.fit(X,Y,batch_size=32, epochs = 10, validation_split=0.1,callbacks=[tensorboard])