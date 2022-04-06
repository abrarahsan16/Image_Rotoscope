import numpy as np
import sys
import argparse
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from cartoon_effect.cartooner import stylization

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
    imgStyle = stylization(img, imgGrey, 12, 5, 8, 5)
    #imgStyle = imgStyle*255
    imgStyle = (imgStyle/imgStyle.max())*255
    if img.ndim == 3:
        imgS = np.zeros(img.shape)
        imgS[:, :, 0] = imgStyle
        imgS[:, :, 1] = imgStyle
        imgS[:, :, 2] = imgStyle
        imgOutput = imgS+img
        imgOutput = (imgOutput/imgOutput.max())*255
    elif img.ndim == 2:
        imgS = np.zeros(img.shape)
        imgS = imgStyle
        imgOutput = imgS + img
        imgOutput = (imgOutput/imgOutput.max())*255
        
    return (imgOutput.astype(np.uint8))

if __name__ == '__main__':
    #if (len(sys.argv)<2):
    #    raise Exception("Invalid number of arguments")
    #else:
    #    print('This need to be implemented!')
    img = imread("E:\Github\Image_Rotoscope\samples\\building.jpg", as_gray = False)
    outputImg = cartoonizer(img)
    imsave("E:\Github\Image_Rotoscope\Output Folder\output.jpg", (outputImg))

    print("New image have been saved in the output folder!")