import numpy as np
import sys
import argparse
from skimage.io import imread, imsave
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

def cartoonizer():
    converter = CartoonEffect()

if __name__ == '__main__':
    if (len(sysm.argv)<2):
        raise Exception("Invalid number of arguments")
    else:
        print('This need to be implemented!')