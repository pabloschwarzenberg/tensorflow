from tensorflow import keras
import tensorflow as tf
import os

sess = tf.Session()
model = tf.saved_model.load(sess,['serve'],os.getcwd()+"/predictor")

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("dense_input:0")

data=[0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0
,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,
1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,
0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

clasificador = graph.get_tensor_by_name("dense_1/BiasAdd:0")
predicciones=sess.run(clasificador, feed_dict={x: [data]})

print(predicciones)