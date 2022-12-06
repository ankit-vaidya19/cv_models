import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

class Cifar10():
  def __init__(self):
    (self.x_train,self.y_train),(self.x_test,self.y_test) = cifar10.load_data()
    self.x_train = self.x_train.astype('float32')/255.0
    self.x_test = self.x_test.astype('float32')/255.0

  def split(self,split_size):
    split = int(split_size*len(self.x_train))
    self.x_train1 = self.x_train[:split]
    self.x_val = self.x_train[split:]
    self.y_train1 = self.y_train[:split]
    self.y_val = self.y_train[split:]
    print(self.x_train1.shape)
    print(self.x_val.shape)
    print(self.y_train1.shape)
    print(self.y_val.shape)

  def show_samples(self):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(self.x_train1[i])
        plt.xlabel(class_names[self.y_train1[i][0]])
    plt.show()

