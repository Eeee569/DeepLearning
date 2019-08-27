import cv2
import tensorflow as tf

CATEGORIES = ["Dog","Cat"]

def prepare(filepath):
    IMG_SIZE = 70
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)


model = tf.keras.models.load_model("model_name.model")
prediction = model.predict([prepare('img.jpg')])
print(prediction)