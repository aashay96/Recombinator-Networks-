import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time


FTRAIN = './training.csv'

def load(test=False, cols=None):

    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]
    df = df.dropna()

    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 96, 96, 1)
    return X, y

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X, y = load2d();

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y[i], ax)

plt.show()

def repeat_elements(x, rep, axis):

    x_shape = x.get_shape().as_list()
    splits = tf.split(x, x_shape[axis], axis)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(x_rep,axis)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class rcn():

    def max_pool(self,bottom,name):
            return tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides =[1,2,2,1],padding= 'VALID',name = name)

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
        #bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        #biases = tf.get_variable(bias_init_val,initializer=tf.contrib.layers.xavier_initializer(), name='b')
        bias_init_val = tf.contrib.layers.xavier_initializer_conv2d()
        biases = tf.Variable(bias_init_val(shape=[n_out]), name='b')
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
        net["conv1_3"] = self.conv_layer(net["conv1_4_pool"], name="conv1_3",  n_out=32)
        net["conv1_3_pool"] = self.max_pool(net["conv1_3"],"conv1_4")
        net["conv1_2"] = self.conv_layer(net["conv1_3_pool"], name="conv1_3",  n_out=48)
        net["conv1_2_pool"] = self.max_pool(net["conv1_2"],"conv1_4")
        net["conv1_1"] = self.conv_layer(net["conv1_2_pool"], name="conv1_3",  n_out=48)

        net["conv2_1"]=self.conv_layer(net["conv1_2_pool"], name="conv2_1", n_out=48)
        net["conv3_1"]=self.conv_layer(net["conv2_1"], name="conv3_1", n_out=48)
        net["conv4_1"] = self.upsample(net["conv1_1"], [2,2])
        concat = tf.concat([net["conv4_1"],net["conv1_2"]],3) #n_out 96,K feature Map
        net["conv2_2"]=self.conv_layer(concat, name="conv2_2", n_out=48)
        net["conv3_2"]=self.conv_layer(net["conv2_2"], name="conv2_3", n_out=32)
        net["conv4_2"] = self.upsample(net["conv3_2"], [2,2])

        concat1 = tf.concat( [net["conv4_2"], net["conv1_3"]],3) #n_out 64,K feature Map
        net["conv2_3"]=self.conv_layer(concat1, name="conv2_3", n_out=32)
        net["conv3_3"]=self.conv_layer(net["conv2_3"], name="conv3_3", n_out=16)
        net["conv4_3"] = self.upsample(net["conv3_3"], [2,2])

        concat2 = tf.concat( [net["conv4_3"], net["conv1_4"]],3) #n_out 32,K feature Map
        net["conv2_4"]=self.conv_layer(concat2, name="conv2_4", n_out=16)
        net["conv3_4"]=self.conv_layer(net["conv2_2"], name="conv3_4", n_out=5)

        shp = net["conv3_4"].get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        flat = tf.reshape(net["conv3_4"], [-1, flattened_shape], name="flat")
        flat = tf.nn.dropout(flat,dropout_keep_prob, name='flat_drop')
        net["fc1"] = self.fc_op(flat, name="fc1", n_out=500)
        net["fc1_drop"] = tf.nn.dropout(net["fc1"], 0.9, name="fc1_drop")
        net["fc2"] = self.fc_op(net["fc1_drop"], name="fc2", n_out=30)
        return net




channels = 1
width, height = [96,96]
batch_size = 100
num_epochs = 100
lr=0.01
tf.reset_default_graph()
in_images = tf.placeholder("float", [batch_size, width, height, channels])
position = tf.placeholder("float", [batch_size,y.shape[1]])
model = rcn()
net = model.build(in_images)
last_layer = net["fc2"]
loss = tf.reduce_mean(tf.square(last_layer - position))
#loss = tf.reduce_mean(-tf.reduce_sum(position * tf.log(last_layer), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(lr)
#global_step = tf.Variable(0, name="global_step", trainable=False)
train_step = optimizer.minimize(loss)
initializer = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())
with tf.Session() as sess:
    sess.run(initializer)
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
            inputs, targets = batch
            result = sess.run(
                [train_step, loss],
                feed_dict = {
                    in_images: inputs,
                    position: targets,
                }
            )
            #print result
            train_err += result[1]
            train_batches += 1

            if np.isnan(result[1]):
                print("gradient vanished/exploded")
                break
            print '=',
        # Then we print the results for this epoch:
        print("\nEpoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if epoch%10 == 0:
            checkpoint_path = saver.save(sess, "model.ckpt")
