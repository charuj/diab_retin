'''
 1) Making the input images the same dimensions.
 Note: TF's support for different array shapes in the placeholder is still pretty preliminary.
        You could use the sparse placeholder, but I choose instead to resize the images to the same shape.
2) Generating shuffled batches of images and their labels

'''

import cPickle as pickle
from numpy import genfromtxt
from PIL import Image, ImageOps
import tensorflow as tf


#LABELS
image_labels = genfromtxt('trainLabels2.csv', delimiter=',')  # convert CSV into array
image_labels= image_labels[1:, :]
sample_labels= image_labels[:10,1] # the first 10 are the samples; for entire dataset comment this out

# Loading array of images
list_images_array= pickle.load( open( "file_images_array_processed.p", "rb" ) )
image_size= (64,64) # change this as needed; should be a # divisible by 2 many times



def resize(list_images_array, image_size):
    '''

    Resize all the images to have the same size

    :param list_images_array:
    :return: list of resized images as array
    '''

    resized_image_list=[]

    for image  in list_images_array:
        # FIGURED out what the bug is: the ImageOps fit functions gets the shape of image by calling image.size[0] INSTEAD of image.shape[0]
        # resized_img= ImageOps.fit(image, image_size, Image.ANTIALIAS)
        resized_img= tf.image.resize_image_with_crop_or_pad(tf.convert_to_tensor(image), 64,64)
        resized_image_list.append(resized_img)

    return resized_image_list


def generate_image_and_label_batch(resized_image_list, labels, min_queue_examples,batch_size, shuffle):
    '''

    :param resized_image_list: list of 3-D tensors of [height, width, 3] of type.float32
    :param labels: list of 1-D tensors of type.int32
    :param min_queue_examples: minimum number of samples to keep in the queue that provides the batches of examples, int32
    :param batch_size: number of images per batch, int
    :param shuffle: boolean that indicates whether to use shuffling in the queue
    :return:
            batch_images: 4D tensor of size [batch_size, height, width, 3]
            labels: 1D tensor of size [batch_size]

    '''

    #1. Create a queue to shuffle examples
    #2. Read, in intervals of batch_size, image + labels from the example queue


    num_preprocess_threads= 16 # what is this?
    resized_image_list=tf.to_float(resized_image_list)
    if shuffle:
        images_batch, labels_batch= tf.train.shuffle_batch([resized_image_list, labels],
                                                           batch_size=batch_size,
                                                           capacity=min_queue_examples + 3*batch_size,
                                                           num_threads=num_preprocess_threads,
                                                           min_after_dequeue=min_queue_examples)
    else:
        images_batch, labels_batch= tf.train.batch([resized_image_list, labels],
                                                   batch_size= batch_size,
                                                   num_threads=num_preprocess_threads,
                                                   capacity= min_queue_examples + 3*batch_size)
    #Display training images
    tf.image_summary('images', images_batch)

    return images_batch, tf.reshape(labels_batch,[batch_size])




## MAIN ##

resized_image_list= resize(list_images_array, image_size)
image_batch, label_batch= generate_image_and_label_batch(resized_image_list[0], sample_labels, 2, 2, shuffle=True)
