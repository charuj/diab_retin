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
import os.path
from os.path import isfile, join
import cPickle as pickle

cropped_images= pickle.load( open( "file_cropped.p", "rb" ) )


#CROPPING BORDERS

def trim(directory):
    '''

    :param im: original retinal image that is to be cropped (the black border in particular)
    :return: a list of 3D arrays, where each array corresponds to one cropped image. The third dimension is the RGB channel
            and will be 3.
            This function doesn't crop every trace of border, it leaves some as a safe measure against over-cropping
    '''

    list_arrays= []

    for item in os.listdir(directory):
        if not item.startswith('.')  and isfile(join(directory,item)):  # to get rid of the hidden file ".DS_Store"; http://stackoverflow.com/questions/15235823/how-to-ignore-hidden-files-in-python-functions
        # if item.endswith('.jpeg') and isfile(join(directory,item)):
            im= Image.open(directory+ item)
            im2= im.crop(im.getbbox())
            arr = np.array(im2)
            list_arrays.append(arr)
    return list_arrays
'''
def trim_raw(images_raw):

    #SAME AS ABOVE, but don't use this one
    list_arrays= []

    for item in images_raw:
        im= Image.open(item)
        im2= im.crop(im.getbbox())
        arr = np.array(im2)
        list_arrays.append(arr)
    return list_arrays
'''



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
    NOTE: Don't need this anymore, as it's replaced by the cropping function
    '''
    list_arrays=[]
    for item in os.listdir(directory):
        if not item.startswith('.') and isfile(join(directory,item)):  # to get rid of the hidden file ".DS_Store"; http://stackoverflow.com/questions/15235823/how-to-ignore-hidden-files-in-python-functions
            img = Image.open()
            arr = np.array(img)  # at this point each individual array will be 3D
            list_arrays.append(arr)

    return list_arrays


def compute_pca(list_cropped_images):
    '''ALTERING INTENSITIES OF RGB CHANNELS
    From Krizhevsky et al. 2012: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    perform PCA on the set of RGB pixel values throughout the training set
    "PCA could approximately capture an important property of natural images, namely, that object identity is invariant to
    changes in the intensity and color of the illumination

    :param list_cropped_images: the output of trim, which is a list of arrays corresponding to the images in dimension width x height x 3 (rgb)
    :return: 2 lists:
            - list of eigenvalues, where each element in the list belongs to one image; shape= Number_of_images x 3 x 1
            - list of eigenvectors, where each element belongs to one image; shape = Number_of_images x 3 x 3


    '''

    eigenvalues_list=[]
    eigenvectors_list= []

    for item in list_cropped_images:

        #Reshape the original image array from height x width x channels ---> to (height*width)x channels
        #img = Image.open(item)
        arr = item
        reshaped_array= arr.reshape([arr.shape[0]*arr.shape[1], arr.shape[2]])

        # Get covariance matrix, eigenvectors, and eigenvalues
        cov= np.dot(reshaped_array.T, reshaped_array) # makes a 3X3 covariance matrix
        eigenvectors,S,V= np.linalg.svd(cov)
        eigenvectors_list.append(eigenvectors)
        eigenval= np.sqrt(S) # cov is symmetric and positive semi-definite
        eigen_dim1= eigenval.shape[0]
        eigenvalues= eigenval.reshape(eigen_dim1, 1)
        eigenvalues_list.append(eigenvalues)

    return eigenvalues_list, eigenvectors_list


def add_colour_noise(list_cropped_images, eigenvalues_list, eigenvectors_list, mu=0.0, sigma=0.1):
    '''

    :param list_cropped_images: the output of trim, which is a list of arrays corresponding to the images in dimension width x height x 3 (rgb)
    :param eigenvalues_list: output of compute_pca, list of eigevalues, where each element in list corresponds to an image
    :param eigenvectors_list: output of compute_pca, list of eigenvators, where each element in list corresponds to an image
    :param mu: mean, set to 0
    :param sigma: std dev, set to 0.1
    :return: a list of adjusted images in 3 dimension, where each element in the list corresponds to one image. Shape of list= 
    '''


    # Generate alpha, which is a random variable drawn from a Gaussian with mean 0 and std dev 0.1
    # Alpha is drawn only once for all the pixels of a particular image in a round of training
    # one alpha value per RGB (i.e. 3 alphas)
    # for i in xrange(number of samples) ### fix this later, should loop over the number of samples in the training set (i.e. num images)

    list_adjusted_img=[]

    for i in range(len(list_cropped_images)):

        alphas= np.random.normal(mu, sigma, [3,3]) # 3 for R,G,B
        augmentation= np.dot(alphas, eigenvalues_list[i])
        noise= np.dot(eigenvectors_list[i], augmentation)

        # add noise to the image array
        adjusted_img= list_cropped_images[i] + noise.T # gives original dimension (this is how to check if it works)
        list_adjusted_img.append(adjusted_img)

    return list_adjusted_img




## MAIN ##

# list_arrays= get_image_arrays("/Users/charujaiswal/PycharmProjects/diab_retin/sample")
list_im_arrays= trim("/Users/charujaiswal/PycharmProjects/diab_retin/sample/")
eigenvalues_list, eigenvectors_list= compute_pca(list_im_arrays)
list_adjusted_img= add_colour_noise(list_im_arrays, eigenvalues_list, eigenvectors_list, mu=0.0, sigma=0.1)


