import numpy as np
import sys
import argparse
from skimage.io import imread, imsave
from skimage.color import rgb2grey
from cartoon_effect.cartooner import CartoonEffect
'''Project Submission
Abrar Ahsan, Mahnoor Hussain
500722182, 500837367
Required Parameters
----------
inputPath : str
    input image file name and path

outputPath : str
    output image file name and path

More inputs will be needed here.
'''

def cartoonizer(img):
    converter = CartoonEffect()
    if img.ndim != 3 and img.ndim !=2:
        raise ValueError("Image needs to be RGB or greyscale")
    elif img.ndim == 3:
        imgGrey = rgb2grey(img)
    else:
        imgGrey = img
    
    imgOutput = converter.stylization()

    return imgOutput

if __name__ == '__main__':
    if (len(sys.argv)<2):
        raise Exception("Invalid number of arguments")
    else:
        print('This need to be implemented!')

    outputImg = cartoonizer(img)

    print("New image have been saved in the output folder!")