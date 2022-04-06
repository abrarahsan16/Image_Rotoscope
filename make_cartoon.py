import numpy as np
import sys
import argparse
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from cartoon_effect.cartooner import stylization
from scipy.ndimage import histogram
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
    if img.ndim != 3 and img.ndim !=2:
        raise ValueError("Image needs to be RGB or greyscale")
    elif img.ndim == 3:
        imgGrey = rgb2gray(img)
    else:
        imgGrey = img
    #, img, imgGrey, sigmaBlur, row, N)
    imgStyle = stylization(img, imgGrey, 7, 2.28, 3, 3)
    #imgOutput = imgStyle*img
    #imgOutput = (imgOutput/imgOutput.max())*255

    #imgOutput = (imgOutput/imgOutput.max())*255

    return (imgStyle)

if __name__ == '__main__':
    #if (len(sys.argv)<2):
    #    raise Exception("Invalid number of arguments")
    #else:
    #    print('This need to be implemented!')
    img = imread("E:\Github\Image_Rotoscope\samples\\building.jpg", as_gray = False)
    outputImg = cartoonizer(img)
    imsave("E:\Github\Image_Rotoscope\Output Folder\output.jpg", (outputImg))

    print("New image have been saved in the output folder!")