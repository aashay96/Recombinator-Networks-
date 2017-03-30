import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time


class rcn():

    def max_pool(self,bottom,name):
            return tf.nn.max_pool(bottom,ksize=[1,2,2,1],stride =[1,2,2,1],padding= 'VALID',name = name)

    def conv_filter(self, name, n_in, n_out):
        """
        kw, kh - filter width and height
        n_in - number of input channels
        n_out - number of output channels
        """
        kernel_init_val = tf.truncated_normal([3, 3, n_in, n_out], dtype=tf.float32, stddev=0.1)
        kernel = tf.Variable(kernel_init_val, trainable=True, name='w')
        return kernel

    def conv_bias(self, name, n_out):
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True,initializer=tf.contrib.layers.xavier_initializer(), name='b')
        return biases



    def conv_layer(self,bottom,name,n_out):
        with tf.variable_scope(name):
            n_in = bottom.get_shape()[-1].value
            #read https://www.tensorflow.org/programmers_guide/variable_scope,weights and biases in downloaded file
            filters = self.conv_filter(name,n_in,n_out)
            conv = tf.nn.conv2d(bottom,filters,[1,1,1,1],padding='SAME')
            conv_biases = self.conv_bias(name,n_out)
            conv = tf.nn.bias_add(conv,conv_biases)
            tanh = tf.nn.tanh(conv)
            return tanh

    def upsample(self,X,scale):
        output = repeat_elements(repeat_elements(X, scale[0], axis=1),scale[1], axis=2)
        return output

    def fc_op(self, input_op, name, n_out):
        n_in = input_op.get_shape()[-1].value

        with tf.variable_scope(name):
            kernel = tf.Variable(tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=0.1), name='w')
            biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), name='b')
            activation = tf.nn.tanh(tf.matmul(input_op, kernel) + biases)
            return activation

    def build(self,input_im,dropout_keep_prob=0.75):  
        net = {}
        net["conv1_4"] = self.conv_layer(input_im, name="conv1_4", n_out=16)
        net["conv1_4_pool"] = self.max_pool(net["conv1_4"],"conv1_4")
        net["conv1_3"] = self.conv_layer(net["conv1_4"], name="conv1_3",  n_out=32)
        net["conv1_3_pool"] = self.max_pool(net["conv1_3"],"conv1_4")
        net["conv1_2"] = self.conv_layer(net["conv1_3"], name="conv1_3",  n_out=48)
        net["conv1_2_pool"] = self.max_pool(net["conv1_2"],"conv1_4")
        net["conv1_1"] = self.conv_layer(net["conv1_2"], name="conv1_3",  n_out=48)

        net["conv2_1"]=self.conv_layer(net["conv1_1"], name="conv2_1", n_out=48)
        net["conv3_1"]=self.conv_layer(net["conv1_1"], name="conv2_1", n_out=48)
        net["conv4_1"] = self.upsample(net["conv3_1"], [2,2])
        concat = tf.concat(3, [net["conv4_1"], net["conv1_2"]]) #n_out 96,K feature Map
        net["conv2_2"]=self.conv_layer(concat, name="conv2_2", n_out=48)
        net["conv3_2"]=self.conv_layer(net["conv2_2"], name="conv2_3", n_out=32)
        net["conv4_2"] = self.upsample(net["conv3_2"], [2,2])

        concat1 = tf.concat(3, [net["conv4_2"], net["conv1_3"]]) #n_out 64,K feature Map
        net["conv2_3"]=self.conv_layer(concat1, name="conv2_3", n_out=32)
        net["conv3_3"]=self.conv_layer(net["conv2_3"], name="conv3_3", n_out=16)
        net["conv4_3"] = self.upsample(net["conv3_3"], [2,2])

        concat = tf.concat(3, [net["conv4_3"], net["conv1_4"]]) #n_out 32,K feature Map
        net["conv2_4"]=self.conv_layer(concat, name="conv2_4", n_out=16)
        net["conv3_4"]=self.conv_layer(net["conv2_2"], name="conv3_4", n_out=5)

        shp = net["conv3_4"].get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        flat = tf.reshape(net["conv3_4"], [-1, flattened_shape], name="flat")
        flat = tf.nn.dropout(flat,dropout_keep_prob, name='flat_drop')
        net["fc1"] = self.fc_op(flat, name="fc1", n_out=500)
        net["fc1_drop"] = tf.nn.dropout(net["fc1"], 0.9, name="fc1_drop")
        net["fc2"] = self.fc_op(net["fc1_drop"], name="fc2", n_out=30)
        return net
