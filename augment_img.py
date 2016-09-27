'''

Additional preprocessing/augmentation than what is in image_processing.py

What's included:

1. Cropping border: uses PIL's built-in bounding-box detection
2.

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
    return im2


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


def rgb_pca(img):
    '''ALTERING INTENSITIES OF RGB CHANNELS
    From Krizhevsky et al. 2012: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    perform PCA on the set of RGB pixel values throughout the training set
    "PCA could approximately capture an important property of natural images, namely, that object identity is invariant to
    changes in the intensity and color of the illumination '''

    # This function also flattens the images (is a replacement for image_processing.py)


list_arrays= get_image_arrays("/Users/charujaiswal/PycharmProjects/diab_retin/sample")



