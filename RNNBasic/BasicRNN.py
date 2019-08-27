import tensorflow as tf
Sequential = tf.keras.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
LSTM = tf.keras.layers.LSTM
cuLSTM = tf.keras.layers.CuDNNLSTM

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255





config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


model = Sequential()

model.add(cuLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))

model.add(cuLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(10,activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test))
