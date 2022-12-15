import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plot
from dataset import *


cifar10 = Cifar10()


cifar10.split(0.9)


def identity_block(x,f,filters):
    F1,F2,F3 = filters
    x_skip = x
    x = layers.Conv2D(filters = F1,kernel_size = (1,1),strides = (1,1),padding = 'valid')(x)
    x = layers.BatchNormalization()(x,training = True)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = F2,kernel_size = (f,f),strides = (1,1),padding = 'same')(x)
    x = layers.BatchNormalization()(x,training = True)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = F3,kernel_size = (1,1),strides = (1,1),padding = 'valid')(x)
    x = layers.BatchNormalization()(x,training = True)

    x = layers.Add()([x,x_skip])
    x = layers.Activation('relu')(x)
    return x


def conv_block(x,f,filters,s = 2):
    F1,F2,F3 = filters
    x_skip = x
    x = layers.Conv2D(filters = F1,kernel_size = (1,1),strides = (s,s),padding = 'valid')(x)
    x = layers.BatchNormalization()(x,training = True)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = F2,kernel_size = (f,f),strides = (1,1),padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = F3,kernel_size = (1,1),strides = (1,1),padding = 'valid')(x)
    x = layers.BatchNormalization()(x)
    
    x_skip = layers.Conv2D(filters = F3,kernel_size = (1,1),strides = (s,s),padding = 'valid')(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)

    x = layers.Add()([x,x_skip])
    x = layers.Activation('relu')(x)
    return x


inputs = keras.Input(shape = (32,32,3))
x = layers.ZeroPadding2D((3,3))(inputs)
x = layers.Conv2D(64,kernel_size = (7,7),strides = (2,2))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size = (3,3),strides = (2,2))(x)

x = conv_block(x,3,filters = [64,64,256],s = 1)
x = identity_block(x,3,[64,64,256])
x = identity_block(x,3,[64,64,256])

x = conv_block(x,3,[128,128,512],2)
x = identity_block(x,3,[128,128,512])
x = identity_block(x,3,[128,128,512])
x = identity_block(x,3,[128,128,512])

x = conv_block(x,3,[256,256,1024],2)
x = identity_block(x,3,[256,256,1024])
x = identity_block(x,3,[256,256,1024])
x = identity_block(x,3,[256,256,1024])
x = identity_block(x,3,[256,256,1024])
x = identity_block(x,3,[256,256,1024])

x = conv_block(x,3,[512,512,2048],2)
x = identity_block(x,3,[512,512,2048])
x = identity_block(x,3,[512,512,2048])

x = layers.AveragePooling2D(pool_size = (1,1))(x)

x = layers.Flatten()(x)
output = layers.Dense(10,activation = 'softmax')(x)

model = keras.Model(inputs = inputs,outputs = output)


model.summary()


model.compile(optimizer = keras.optimizers.SGD(learning_rate = 0.1,momentum = 0.9,),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
              metrics = ['accuracy'])


lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',factor = 0.1,patience = 5,mode = 'max',cooldown = 0)


history = model.fit(cifar10.x_train1,cifar10.y_train1,batch_size = 256,epochs = 100,callbacks = [lr_reducer],validation_data = (cifar10.x_val,cifar10.y_val))


history1 = model.evaluate(cifar10.x_test,cifar10.y_test,batch_size = 256)


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