

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



def batch(data, labels, batch_size, num_epochs, shuffle=True):
    data_size= len(data)
    num_batches_per_epoch= int(data_size/ batch_size)
    for epoch in range(num_epochs):
        if shuffle:
            np.random.shuffle([data, labels])
        else:
            data= data
            labels= labels
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index], labels[start_index:end_index]



batch(list_images_array, sample_labels, 5, 10, shuffle=True)



