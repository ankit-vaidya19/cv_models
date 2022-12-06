import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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


print(x_train1.shape)
print(y_train1.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)


model = keras.Sequential([
    layers.Input(shape=(32,32,3)),
    
    layers.Conv2D(64,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.Conv2D(64,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.MaxPooling2D((2,2),strides = (2,2)),
    
    layers.Conv2D(128,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.Conv2D(128,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.MaxPooling2D((2,2),strides = (2,2)),

    layers.Conv2D(256,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.Conv2D(256,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.Conv2D(256,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.MaxPooling2D((2,2),strides = (2,2)),

    layers.Conv2D(512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.Conv2D(512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.Conv2D(512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.MaxPooling2D((2,2),strides = (2,2)),

    layers.Conv2D(512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.Conv2D(512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.Conv2D(512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
    layers.MaxPooling2D((2,2),strides = (2,2)),

    layers.Flatten(),
    
    layers.Dense(4096,activation = 'relu'),
    layers.Dropout(0.5),
    
    layers.Dense(4096,activation = 'relu'),
    layers.Dropout(0.5),
    
    layers.Dense(10,activation = 'softmax')
])


lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',factor = 0.1,patience = 5,mode = 'max',cooldown = 0)
early_stopper = tf.keras.callbacks.EarlyStopping(monitor = 'accuracy',patience = 3,mode = 'auto')


model.compile(optimizer = keras.optimizers.SGD(learning_rate = 0.01,momentum = 0.9,),loss = keras.losses.categorical_crossentropy,metrics = ['accuracy'])


model.summary()


history = model.fit(x_train1,y_train1,batch_size = 256,epochs = 40,validation_data = (x_val,y_val),callbacks = [lr_reducer,early_stopper])


history1 = model.evaluate(x_test,y_test,batch_size = 256)


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