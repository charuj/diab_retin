import numpy as np
from PIL import Image


'''

Mac terminal script to reduce image sizes:

-get to the directory where the images are (using cd)

sips -Z 100 *.jpeg

- sips is the command being used
-Z tells it to maintain the aspect ratio
- 100 is the new size of the image
- *.jpeg instructs the computer to downsize every image ending in jpeg

'''


# List all files of the directory containing the images
import os
from os import listdir
from os.path import isfile, join



cwd = os.getcwd()  # Get the current working directory (cwd)
newcwd = os.chdir("/Users/charujaiswal/PycharmProjects/diab_retin/sample")
files = os.listdir("/Users/charujaiswal/PycharmProjects/diab_retin/sample")  # Get all the files in that directory
print("Files in '%s': %s" % (newcwd, files))


root= "/Users/charujaiswal/PycharmProjects/diab_retin/sample"

images_list= []
for item in os.listdir("/Users/charujaiswal/PycharmProjects/diab_retin/sample"):
    if not item.startswith('.') and isfile(join(root, item)): # to get rid of the hidden file ".DS_Store"; http://stackoverflow.com/questions/15235823/how-to-ignore-hidden-files-in-python-functions
        img= Image.open(item)
        arr = np.array(img) # at this point each individual array will be 3D
        pixels= arr.flatten()
        images_list.append(pixels)

images_array= np.array(images_list)
print images_array.shape()
#
# # Convert images to pixel and rgb values
# img = Image.open(onlyfiles[2])
# arr= np.array(img)
# print arr.shape()



