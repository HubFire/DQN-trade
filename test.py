import numpy as np
import talib 
import  tensorflow as tf 

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

a = tf.constant([1,2,3],shape=[3,1])
b = tf.constant([0,1,0],shape=[3,1])
op_mul = tf .multiply(a,b)

c = tf.reduce_sum(op_mul,reduction_indices =1)
print session.run(c)
session.close()





