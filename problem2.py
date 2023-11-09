import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def loadbayer(path):
    """ Load data from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array (H,W)
    """
    # code to load my .npy file and return a numpy array
    data = np.load(path)
    return data
    


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        bayerdata: Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """
    h, w = bayerdata.shape
    red = np.zeros((h, w))
    green = np.zeros((h, w))
    blue = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if i % 2 == 0:
                if j % 2 == 0:
                    red[i, j] = bayerdata[i, j]
                else:
                    green[i, j] = bayerdata[i, j]
            else:
                if j % 2 == 0:
                    green[i, j] = bayerdata[i, j]
                else:
                    blue[i, j] = bayerdata[i, j]

    return red, green, blue
    
    


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        r: red channel as numpy array (H,W)
        g: green channel as numpy array (H,W)
        b: blue channel as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    return np.dstack((r,g,b))
    


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        r: red channel as numpy array (H,W)
        g: green channel as numpy array (H,W)
        b: blue channel as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """
    h, w = r.shape
    interpolated_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Create convolution filters for interpolation
    kernel_r = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    kernel_g = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    kernel_b = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

    # Apply convolution to interpolate each channel
    interpolated_image[:, :, 0] = convolve(r, kernel_r, mode='constant', cval=0)
    interpolated_image[:, :, 1] = convolve(g, kernel_g, mode='constant', cval=0)
    interpolated_image[:, :, 2] = convolve(b, kernel_b, mode='constant', cval=0)

    return interpolated_image



