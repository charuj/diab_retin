

import cPickle as pickle
import numpy as np
from numpy import genfromtxt
from PIL import Image, ImageOps
import tensorflow as tf
import random


#LABELS
image_labels = genfromtxt('trainLabels2.csv', delimiter=',')  # convert CSV into array
image_labels= image_labels[1:, :]
sample_labels= image_labels[:10,1] # the first 10 are the samples; for entire dataset comment this out
sample_labels= np.reshape(sample_labels, [sample_labels.shape[0],1])

# Loading array of images
list_images_array= pickle.load( open( "file_images_array_processed.p", "rb" ) )

def generate_batches(list_images_array, labels, min_queue_examples, batch_size, capacity):
    # images_tensor= tf.sparse_to_dense(list_images_array)
    # images_tensor= tf.sparse_to_dense(list_images_array, )
    # labels_tensors= tf.sparse_to_dense(labels)

    labels= tf.sparse_tensor_to_dense(labels)

    images_batch, labels_batch= tf.train.shuffle_batch([list_images_array, labels], batch_size, capacity=capacity, min_after_dequeue=min_queue_examples)

    return images_batch, tf.reshape(labels_batch, [batch_size])


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data_size= len(data)
    num_batches_per_epoch= int(data_size/ batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            print shuffle_indices
            shuffled_data = data[shuffle_indices]

        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



# images_batch, labels_batch= generate_batches(list_images_array, sample_labels, min_queue_examples=1, batch_size=5, capacity=2)

