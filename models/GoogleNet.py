import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from dataset import *


cifar10 = Cifar10()


cifar10.split(0.8)


def inception_module(x,f1,f2_1,f2_2,f3_1,f3_2,f4):
    x1 = layers.Conv2D(filters = f1,kernel_size = (1,1),strides = (1,1),padding = 'same',activation = 'relu')(x)
    
    x2 = layers.Conv2D(filters = f2_1,kernel_size = (1,1),strides = (1,1),padding = 'same',activation = 'relu')(x)
    x2 = layers.Conv2D(filters = f2_2,kernel_size = (3,3),strides = (1,1),padding = 'same',activation = 'relu')(x2)
    
    x3 = layers.Conv2D(filters = f3_1,kernel_size = (1,1),strides = (1,1),padding = 'same',activation = 'relu')(x)
    x3 = layers.Conv2D(filters = f3_2,kernel_size = (5,5),strides = (1,1),padding = 'same',activation = 'relu')(x3)
    
    x4 = layers.MaxPooling2D((3,3),strides = (1,1),padding = 'same')(x)
    x4 = layers.Conv2D(filters = f4,kernel_size = (4,4),strides = (1,1),padding = 'same',activation = 'relu')(x4)
    
    x = layers.concatenate([x1, x2, x3, x4], axis=-1)
    
    return x


inputs = keras.Input(shape = (32,32,3))
x = layers.Conv2D(filters = 64,kernel_size = (7,7),strides = (2,2),padding = 'valid',activation = 'relu')(inputs)
x = layers.MaxPooling2D((3,3),strides = 2,padding = 'same')(x)
x = layers.Conv2D(filters = 64,kernel_size = (1,1),strides = (2,2),padding = 'same',activation = 'relu')(x)#local response normalisation

x = layers.Conv2D(filters = 192,kernel_size = (3,2),strides = (1,1),padding = 'same',activation = 'relu')(x)
x = layers.MaxPooling2D((3,3),strides = 2,padding = 'same')(x)

x = inception_module(x,64,96,128,16,32,32)
x = inception_module(x,128,128,192,32,96,64)
x = layers.MaxPooling2D((3,3),strides = 2,padding = 'same')(x)
x = inception_module(x,192,96,208,16,48,64)

#first auxiliary classifier
x1 = layers.AveragePooling2D((5,5),strides = 3,padding = 'same')(x)
x1 = layers.Conv2D(filters = 128,kernel_size = (1,1),padding = 'same',activation = 'relu')(x1)
x1 = layers.Flatten()(x1)
x1 = layers.Dense(1024,activation = 'relu')(x1)
x1 = layers.Dropout(0.7)(x1)
x1 = layers.Dense(10,activation = 'softmax',name = 'auxiliary_1')(x1)

x = inception_module(x,160,112,224,24,64,64)
x = inception_module(x,128,128,256,24,64,64)
x = inception_module(x,112,144,288,32,64,64)

#second auxiliary classifier
x2 = layers.AveragePooling2D((5,5),strides = 3,padding = 'same')(x)
x2 = layers.Conv2D(filters = 128,kernel_size = (1,1),padding = 'same',activation = 'relu')(x2)
x2 = layers.Flatten()(x2)
x2 = layers.Dense(1024,activation = 'relu')(x2)
x2 = layers.Dropout(0.7)(x2)
x2 = layers.Dense(10,activation = 'softmax',name = 'auxiliary_2')(x2)


x = inception_module(x,256,160,320,32,128,128)
x = layers.MaxPooling2D((3,3),strides = (2,2),padding = 'same')(x)
x = inception_module(x,256,160,320,32,128,128)
x = inception_module(x,384,192,384,48,128,128)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10,activation = 'softmax',name = 'final_layer')(x)


model = keras.Model(inputs = inputs,outputs = [outputs,x1,x2])
model.summary()


model.compile(optimizer = keras.optimizers.SGD(learning_rate = 0.1,momentum = 0.9,),
              loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
              metrics = ['accuracy'])


def scheduler(epoch,lr):
    rem = epoch % 8
    if rem == 0:
        return lr
    else:
        return lr*0.96
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)


history = model.fit(cifar10.x_train1,[cifar10.y_train1,cifar10.y_train1,cifar10.y_train1],epochs = 50,validation_data=(cifar10.x_val, [cifar10.y_val, cifar10.y_val, cifar10.y_val]),batch_size= 256,callbacks =[lr_scheduler])


history1 = model.evaluate(cifar10.x_test,[cifar10.y_test,cifar10.y_test,cifar10.y_test],batch_size = 256)


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
plt.legend(['train', 'val'], loc='upper right')


