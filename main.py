import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#On importe le datatset
mnist = tf.keras.datasets.mnist

#On divise le datastet entre données de test et données de train

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normamisation
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Mise ne place du model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#Neural Network
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
#Output couche
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Entrainement du model
model.fit(x_train, y_train,epochs=3)

model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

#Evaluer le modele
#loss , accuracy = model.evaluate(x_test, y_test)

#print(loss)
#print(accuracy)

image_number = 1
while os.path.isfile(f"digits/ex{image_number}.png"):
    try:
        img = cv2.imread(f"digits/ex{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Ce nombhre est probablement un {np.argmax(prediction)} ")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except:
        print("Error")

    finally:
        image_number += 1


