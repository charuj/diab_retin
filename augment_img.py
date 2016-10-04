'''

Additional preprocessing/augmentation than what is in image_processing.py

What's included:

1. Cropping border: uses PIL's built-in bounding-box detection
2. Augmenting the RGB channels by doing PCA (variance) a la Krizhevsky et al. 2012
    https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
    

'''

import numpy as np
from PIL import Image
import os
from os.path import isfile, join


#CROPPING BORDERS

def trim(im):
    '''

    :param im: original retinal image that is to be cropped (the black border)
    :return: cropped image in RGB; does not crop every trace of border, leaves some as a safe measure against over-cropping
    '''
    im= Image. open(im)
    im2= im.crop(im.getbbox())
    arr = np.array(im2)
    return arr


'''This version crops too much, better to crop using simple bounding box

def trim(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0,0))) # get border colour from the top left pixel using getpixel; this is so I don't have to pass it the border colour
    diff = ImageChops.difference(img, bg) # subtracts a scalar from the
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
'''

def get_image_arrays(directory):
    '''

    :param directory: directory that contains the training set of images
    :return: a list of 3D arrays, where each array corresponds to an image that is of dimension height x width x RGB channels

    NOTE: this function essentially replaces the file image_processing.py
    '''
    list_arrays=[]
    for item in os.listdir(directory):
        if not item.startswith('.') and isfile(join(directory,item)):  # to get rid of the hidden file ".DS_Store"; http://stackoverflow.com/questions/15235823/how-to-ignore-hidden-files-in-python-functions
            img = Image.open(item)
            arr = np.array(img)  # at this point each individual array will be 3D
            list_arrays.append(arr)
    return list_arrays


def compute_pca(item):
    '''ALTERING INTENSITIES OF RGB CHANNELS
    From Krizhevsky et al. 2012: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    perform PCA on the set of RGB pixel values throughout the training set
    "PCA could approximately capture an important property of natural images, namely, that object identity is invariant to
    changes in the intensity and color of the illumination '''

    # This function also flattens the images

    # for now i'm going to run an image through this and open the image here as well because get_image_arrays is not working

    #Reshape the original image array from height x width x channels ---> to (height*width)x channels
    #img = Image.open(item)
    arr = item
    reshaped_array= arr.reshape([arr.shape[0]*arr.shape[1], arr.shape[2]])

    # Get covariance matrix, eigenvectors, and eigenvalues
    cov= np.dot(reshaped_array.T, reshaped_array) # makes a 3X3 covariance matrix
    eigenvectors,S,V= np.linalg.svd(cov)
    eigenval= np.sqrt(S) # cov is symmetric and positive semi-definite
    eigen_dim1= eigenval.shape[0]
    eigenvalues= eigenval.reshape(eigen_dim1, 1)

    return arr, eigenvalues, eigenvectors


def add_colour_noise(image_array, eigenvalues, eigenvectors, mu=0.0, sigma=0.1):
    # Generate alpha, which is a random variable drawn from a Gaussian with mean 0 and std dev 0.1
    # Alpha is drawn only once for all the pixels of a particular image in a round of training
    # one alpha value per RGB (i.e. 3 alphas)
    # for i in xrange(number of samples) ### fix this later, should loop over the number of samples in the training set (i.e. num images)

    alphas= np.random.normal(mu, sigma, [3,3]) # 3 for R,G,B
    augmentation= np.dot(alphas, eigenvalues)
    noise= np.dot(eigenvectors, augmentation)

    # add noise to the image array
    adjusted_img= image_array + noise.T # gives original dimension (this is how to check if it works)

    return adjusted_img




## MAIN ##

# list_arrays= get_image_arrays("/Users/charujaiswal/PycharmProjects/diab_retin/sample")
im_array= trim("10_left.jpeg")
image_array, eigenvalues, eigenvectors= compute_pca(im_array) # works when the image is in the main directory, not the sample directory
augmented_img= add_colour_noise(image_array, eigenvalues, eigenvectors, mu=0.0, sigma=0.1)


