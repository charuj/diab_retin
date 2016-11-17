'''
[Shallow] Residual Network

- Uses the basic building block of a ResNet from Kaiming He et al.'s paper:
- Also see: http://torch.ch/blog/2016/02/04/resnets.html

The residual block here looks like this:
[ Input > Convolution Layer > Batch normalization > ReLU > Convolution > BN > Addition of Input/Shortcut > RelU > Output  ]

I'm going to be stacking these blocks.

Note: the authors don't using pooling, except for at the beginning to lower the size  of the images a bit
and average pooling before the final layer; I do this as well.


Dependencies:
Numpy
Tensorflow


'''


import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import cPickle as pickle
import augment_img
import input_data

## LOAD DATA
#Labels
image_labels = genfromtxt('trainLabels2.csv', delimiter=',')  # convert CSV into array
image_labels= image_labels[1:, :]
sample_labels= image_labels[:10,1] # the first 10 are the samples; for entire dataset comment this out
sample_labels= np.reshape(sample_labels, [sample_labels.shape[0],1])

# Loading array of images
list_images_array= pickle.load( open( "file_images_array_processed.p", "rb" ) )

# TODO: create validation + test sets

#Parameters
learning_rate= 0.0001
training_iters= 100
batch_size= 5
display_step= 2
epsilon= 1e-3 # small epsilon value for the batch normalization transform


#Network parameters
n_input= augment_img.image_size[0] * augment_img.image_size[1]
n_classes= 5
dropout= 0.5 # prob of keeping a unit
hidden_size = 100 # number of neurons in each hidden layer

# tf Graph input
x= tf.placeholder(tf.float32, [batch_size, augment_img.image_size[0], augment_img.image_size[1], 3], name= 'Images')
y= tf.placeholder(tf.int32, [batch_size, n_classes])
keep_prob= tf.placeholder(tf.float32) # Dropout = keep probability for a neuron




# MAKING WRAPPERS/COMPONENTS OF THE BASIC BLOCK

'''
1) Convolution Wrapper: https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
The conv op sweeps a 2-D filter over batches of images, applying the filter to each window
of each image.

The conv2d op: arbitrary filters that can mix channels together.

Although it's called a convolution op, it's actually "cross correlation" since the filter is combined with an input window
without reversing the filter.

The filter is applied to image patches of the same size as the filter and strided according to the stride argument.

strides = [1, 1, 1, 1] applies the filter to a patch at every offset,
strides = [1, 2, 2, 1] applies the filter to every other image patch in each dimension, etc.

Since input is 4-D, each input[b, i, j, :] is a vector.
For conv2d, these vectors are multiplied by the filter[di, dj, :, :] matrices to produce new vectors.

'''

def conv2d(x, W, b,strides=1):
    '''

    :param x: tensor of type half, float32, or float64
    :param W: filter, must be same type as input
    :param b: a 1-D tensor with size matching x (input)
    :param strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input.
                    Must be in the same order as the dimension specified with format.
    :return: a Conv2D wrapper with bias.
    '''


    x= tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME') # uses tf builtin wrapper
    preactivation= tf.nn.bias_add(x,b) # adding the bias

    return preactivation


def bn(preactivation):
    '''
    Batch normalization using TF builtin
    :param preactivation:
    :return:
    '''

    batch_mean, batch_var= tf.nn.moments(preactivation, [0]) # axes 0
    beta= tf.Variable(tf.zeros(hidden_size))
    gamma= tf.Variable(tf.ones(hidden_size))
    batch_norm= tf.nn.batch_normalization(preactivation, batch_mean, batch_var,beta,gamma, epsilon)
    return batch_norm


def relu(preactivation):
    '''

    :param preactivation: preactivation, that has passed through conv2d
    :return: thresholded (i.e. put through ReLU)
    '''
    return tf.nn.relu(preactivation)


'''
2) Pooling

The pooling opp sweeps a rectangular window over the input tensor, computing a reduction operation for each
window (e.g. average, max). Each pooling op uses rectangular windows of size ksize separated by offset strides.
e.g. if strides is all ones, every window is used. If strides is all twos, every other window is used in each dimension.


'''

def avg_pool(x, k=2 ):
    '''

    :param x: input tensor, which is the output of the conv2d function, should be type float32
    :param ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    :param strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
    :return: A Tensor with type tf.float32. The avg pooled output tensor
    '''

    pooled= tf.nn.avg_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')
    return pooled


def basic_block(x):

    shortcut= x
    with tf.variable_scope("conv1"):

        weights= tf.get_variable('w', [5,5,1,64], tf.float32, tf.contrib.layers.xavier_initializer_conv2d()) # 5x5 conv, 1 input, 64 outputs
        bias = tf.get_variable('b', [64], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
        conv= conv2d(x, weights, bias, strides=1)
        batch_norm= bn(conv)
        relu1= relu(batch_norm)
    with tf.variable_scope("conv2"):
        weights= tf.get_variable('w', [5,5,1,64], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
        bias = tf.get_variable('b', [64], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
        conv= conv2d(relu1, weights, bias, strides=1)
        batch_norm= bn(conv)
        add_shortcut= tf.add(batch_norm, shortcut)
        return relu(add_shortcut)




