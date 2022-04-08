from urllib.request import parse_keqv_list
import numpy as np
import sys
import argparse
from skimage.io import imread, imsave
from skimage.color import rgb2gray
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

Optional Parameters
----------
Sigma : float
    Sigma for the gaussian

SigmaX : Float
    Sigma for the XDoG

P: Float
    P for the XDoG

Epsilon: Float
    Threshold Operator for the XDoG

N: Int
    Number of layers

Returns
-------
Saved image
'''

def cartoonizer(img, sigmaBlur, sigmaDoG, p, epsilon, N):
    # Ensure that the image is 3d or 2d.
    if img.ndim != 3 and img.ndim !=2:
        raise ValueError("Image needs to be RGB or greyscale")
    # If the image is in RGB, convert to greyscale or keep as is
    if img.ndim == 3:
        imgGrey = rgb2gray(img)
    else:
        imgGrey = img
    cartooner = CartoonEffect()
    imgStyle = cartooner.stylization(img, imgGrey, sigmaBlur, sigmaDoG, p, epsilon, N)
    # Normalize the data before adding to the original image and renormalizing
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
    if (len(sys.argv)<2):
        raise Exception("Invalid number of arguments")
    else:
        # Pass the arguments into the parser
        argParser = argparse.ArgumentParser()
        argParser.add_argument("inputPath", type=str)
        argParser.add_argument("outputPath", type=str)
        argParser.add_argument("--sigma", type=float, default = 12)
        argParser.add_argument("--sigmaX", type=float, default = 5)
        argParser.add_argument("--p", type=float, default = 8)
        argParser.add_argument("--epsilon", type=float, default = 0.5)
        argParser.add_argument("--N", type=int, default = 4)
        # Parse the arguments
        args = argParser.parse_args()
        # Read in the image
        img = imread(args.inputPath, as_gray=False)
        outputImg = cartoonizer(img, args.sigma, args.sigmaX, args.p, args.epsilon, args.N)
        imsave(args.outputPath, outputImg)
    
    print("New image have been saved in the output folder!")
    '''
    building = imread("E:\Github\Image_Rotoscope\samples\\building.jpg", as_gray = False)
    #park = imread("E:\Github\Image_Rotoscope\samples\\park.jpg", as_gray = False)
    #stairs = imread("E:\Github\Image_Rotoscope\samples\\stairs.jpg", as_gray = False)
    buildingImg = cartoonizer(building, 12, 5, 8, 0.5, 50)
    #parkImg = cartoonizer(park, 12, 9, 5, 0.4, 4)
    #stairsImg = cartoonizer(stairs, 12, 5, 8, 0.3, 4)
    imsave("E:\Github\Image_Rotoscope\Report Images\\buildingImg_50N.jpg", (buildingImg))
    #imsave("E:\Github\Image_Rotoscope\Report Images\\parkImg_10.jpg", (parkImg))
    #imsave("E:\Github\Image_Rotoscope\Report Images\\stairImg_10.jpg", (stairsImg))
    '''
