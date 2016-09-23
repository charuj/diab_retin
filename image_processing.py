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


# List all files of a directory
from os import listdir
from os.path import isfile, join
onlyfiles= [f for f in listdir("sample") if isfile(join("sample", f))]



