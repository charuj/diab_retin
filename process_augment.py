'''

Additional processing than what is in image_processing.py

What's included:

1. Cropping border from image; note this only really works if the image has a very diff colour than the border. In this case it works


'''

import numpy as np

from PIL import Image, ImageChops

def trim(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0,0))) # get border colour from the top left pixel using getpixel; this is so I don't have to pass it the border colour
    diff = ImageChops.difference(img, bg) # subtracts a scalar from the
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)


img= Image.open("10_left.jpeg")
img= trim(img)
img.show()