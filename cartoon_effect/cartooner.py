import numpy as np
from scipy import ndimage
import math
from skimage.color import rgb2grey
from skimage.morphology import label

class CartoonEffect:

    def stylization(self, img, imgGrey, sigmaBlur, row, T, N):
        imgNew = imgGrey
        # Apply gaussian blur on the image
        imgBlur = self.gaussianBlur(imgNew, sigmaBlur)
        # Find the max and minimum intensities of the image
        Imin = np.min(imgBlur)
        Imax = np.max(imgBlur)
        # Set the total number of bins
        bins = np.clip((Imax-Imin)/N*np.arange(N+1) + Imin, 0, 255)
        # Allocate image L with same size as imgBlur
        L = np.zeros(img.shape[:2], dtype=int)
        # fill in L
        for i in range(N):
            if i == N-1:
                mask = np.logical_and(imgBlur >= bins[i], imgBlur < bins[i+1])
            else:
                mask = imgBlur >= bins[i]
            L[mask] = i
        # Allocate image S with same size as imgBlur
        S = np.zeros(img.shape)
        # If original image is RGB or Grey
        C = np.zeros(img.shape, dtype = int)
        if img.ndim == 3:
            L = label(L)
            for i in range(255):
                Li = L == i
                Li_Connected = np.stack((Li, Li, Li), axis = 2)
                mean_Colour = np.mean(imgBlur, where = Li_Connected, axis=(0, 1))
                C[Li] = mean_Colour
        else:
            C = C/N
        # Return output
        return C * self.XDog(img, sigmaBlur, row, T)

    def gaussianBlur(img, imgGrey, sigma):
        '''Filter an image using a Gaussian kernel.
        The Gaussian is implemented internally as a two-pass, separable kernel.

        Note
        ----
        The kernel is scaled to ensure that its values all sum up to '1'.  The
        slight truncation means that the filter values may not actually sum up to
        one.  The normalization ensures that this is consistent with the other
        low-pass filters in this assignment.

        Parameters
        ----------
        img : numpy.ndarray
            a greyscale image
        sigma : float
            the width of the Gaussian kernel; must be a positive, non-zero value

        Returns
        -------
        numpy.ndarray
            the Gaussian blurred image; the output will have the same type as the
            input

        Raises
        ------
        ValueError
            if the value of sigma is negative
        '''
        if sigma <= 0.01:
            raise ValueError('Sigma should be positive.')
        N = max(6*sigma, 3)
        # If the dimension is even, add 1 to make it odd
        if N % 2 == 0:
            N = N + 1
        # Create one dimension kernel of ascending values
        x_kernel = np.arange(0, N, step=1, dtype=np.float32)
        # Apply the gaussian blur equation
        x_kernel = 1/np.sqrt(2*np.pi*sigma**2) * np.exp((-(x_kernel-math.floor(0.5*N))**2)/(2*sigma**2))
        # Ensure that the shape of the array is correct
        x_kernel = x_kernel.reshape((int(len(x_kernel)), 1))
        # Transpose to create the other dimension
        y_kernel = np.transpose(x_kernel)
        y_kernel = y_kernel.reshape((1, int(len(x_kernel))))
        # Find the sum of the kernel values
        kernel_sum = (x_kernel*y_kernel).sum()
        # Do convolutions one dimension at a time
        conv = self.convolve(img, y_kernel)
        conv = self.convolve(conv, x_kernel)
        # Return the normalized output
        return conv/(kernel_sum)
        raise NotImplementedError('Implement this function/method.')

    def convolve(img, kernel):
        '''Convenience method around ndimage.convolve.
        This calls ndimage.convolve with the boundary setting set to 'nearest'.  It
        also performs some checks on the input image.
        Parameters
        ----------
        img : numpy.ndarray
            input image
        kernel : numpy.ndarray
            filter kernel
        Returns
        -------
        numpy.ndarray
            filter image
        Raises
        ------
        ValueError
            if the image is not greyscale
        TypeError
            if the image or filter kernel are not a floating point type
        '''
        if img.ndim != 2:
            raise ValueError('Only greyscale images are supported.')

        if img.dtype != np.float32 and img.dtype != np.float64:
            raise TypeError('Image must be floating point.')

        if kernel.dtype != np.float32 and img.dtype != np.float64:
            raise TypeError('Filter kernel must be floating point.')

        return ndimage.convolve(img, kernel, mode='nearest')

    
    def XDoG(img, sigma, row, T):
        raise NotImplementedError('Implement this function/method.')
    def LineCleanup():
        raise NotImplementedError('Implement this function/method.')