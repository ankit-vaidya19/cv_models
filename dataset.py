import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


(x_train,y_train),(x_test,y_test) = cifar10.load_data()
split = int(0.8*len(x_train))
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

x_train1 = x_train[:split]
y_train1 = y_train[:split]
x_val = x_train[split:]
y_val = y_train[split:]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.show()