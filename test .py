import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
 
 
facion_mnist_train=pd.read_csv(r"G:\work\Data science\Deep learning\CSV\fashion-mnist_train.csv")
facion_mnist_test=pd.read_csv(r"G:\work\Data science\Deep learning\CSV\fashion-mnist_test.csv")
facion_mnist_train.info()
print(facion_mnist_train.head())
print(facion_mnist_train.shape)
print(facion_mnist_test.shape)

x_train=facion_mnist_train.drop("label",axis=1)
y_train=facion_mnist_train["label"]
x_test=facion_mnist_test.drop("label",axis=1)
y_test=facion_mnist_test["label"]
x_train=np.reshape(x_train,(60000,28,28))
x_test=np.reshape(x_test,(10000,28,28))


model=Sequential([
    layers.Flatten(input_shape=[28,28]),
    layers.Dense(100, activation="relu"),
    layers.Dense(75, activation="relu"),
    layers.Dense(10, activation="softmax")
])

print(model.summary())
model.layers
weights,intercept=model.layers[1].get_weights()
print(weights)
print(intercept)

