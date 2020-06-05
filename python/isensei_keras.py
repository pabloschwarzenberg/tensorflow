import tensorflow as tf
import os
from time import time
import random

random.seed(1)
tf.compat.v1.set_random_seed(1)

def loadTrainSet():
  train=[]
  train_x=[]
  train_y=[]
  train_file=open("train.csv","r")
  for linea in train_file:
    linea=linea.strip()
    linea=linea.split(",")
    linea=list(map(int,linea))
    x=linea[0:128]
    train_x.append(x)
    train_y.append(linea[128:])
  train_file.close()
  train.append( (tf.convert_to_tensor(train_x), tf.convert_to_tensor(train_y)) )
  return train

def loadTestSet():
  test=[]
  test_x=[]
  test_y=[]
  test_file=open("test.csv","r")
  for linea in test_file:
    linea=linea.strip()
    linea=linea.split(",")
    linea=list(map(int,linea))
    x=linea[0:128]
    test_x.append(x)
    test_y.append(linea[128:])
  test_file.close()
  test.append( (tf.convert_to_tensor(test_x), tf.convert_to_tensor(test_y)) )
  return test

predictor=tf.keras.Sequential([ tf.keras.layers.Dense(2, activation=tf.nn.tanh, input_shape=(16*8,)),
        tf.keras.layers.Dense(2) ])
predictor.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['categorical_accuracy'])
train_set=loadTrainSet()
test_set=loadTestSet()

history = predictor.fit(train_set[0][0], train_set[0][1],
                    batch_size=64,
                    epochs=250,
                    steps_per_epoch=10,
                    validation_steps=10,
                    validation_data=(test_set[0][0], test_set[0][1]))

#formato saved_model_cli, tensorflow
from tensorflow import keras
keras.experimental.export_saved_model(predictor,os.getcwd()+"/predictor")
