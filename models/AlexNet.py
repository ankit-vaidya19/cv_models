import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from dataset import *


cifar10 = Cifar10()


cifar10.split(0.8)


model = keras.Sequential([
    layers.Input(shape=(32,32,3)),
    
    layers.Conv2D(96,kernel_size = (11,11),strides = (4,4),padding = 'same',activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2),strides = (2,2),padding = 'same'),
    
    layers.Conv2D(256,kernel_size = (5,5),strides = (1,1),padding = 'same',activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2),strides = (2,2),padding = 'same'),
    
    layers.Conv2D(384,kernel_size = (3,3),strides = (1,1),padding = 'same',activation = 'relu'),
    layers.BatchNormalization(),
    
    layers.Conv2D(384,kernel_size = (3,3),strides = (1,1),padding = 'same',activation = 'relu'),
    layers.BatchNormalization(),
    
    layers.Conv2D(256,kernel_size = (3,3),strides = (1,1),padding = 'same',activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2),strides = (2,2),padding = 'same'),
    
    layers.Flatten(),
    
    layers.Dense(4096,input_shape = (32,32,3),activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(4096,activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(1000,activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(10,activation = 'softmax')
])


model.compile(optimizer = 'Adam',loss = keras.losses.categorical_crossentropy,metrics = ['accuracy'])


model.summary()


history = model.fit(cifar10.x_train1,cifar10.y_train1,batch_size = 32,epochs = 20,validation_data = (cifar10.x_val,cifar10.y_val))


history1 = model.evaluate(cifar10.x_test,cifar10.y_test,batch_size = 32)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')



