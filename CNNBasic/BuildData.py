import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "../Data/kagglecatsanddogs_3367a/PetImages"
CATEGORIES = ["Dog","Cat"]
IMG_SIZE = 50


training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)#0 for dog, 1 for cat
        for img in os.listdir(path):
            try:#some images are broken
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # plt.imshow(img_array, cmap="gray")
                # plt.show()

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()




random.shuffle(training_data)

X = []
Y = []

for features,label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE,1)

pickle_out = open("../Data/X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("../Data/Y.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close()