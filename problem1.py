import numpy as np
import matplotlib.pyplot as plt


################################################################
#             DO NOT EDIT THIS HELPER FUNCTION                 #
################################################################

def load_image(path):
    return plt.imread(path)

################################################################

def display_image(img):
    """ Show an image with matplotlib

    Args:
        img: Image as numpy array (H,W,3)
    """

    #functions needed to display my image, I turned off axis just in case
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    


def save_as_npy(path, img):
    """ Save the image array as a .npy file

    Args:
        img: Image as numpy array (H,W,3)
    """

    #needed to save save image as .npy file
    np.save(path, img)
    


def load_npy(path):
    """ Load and return the .npy file

    Args:
        path: Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    #load my image and store into img variable
    img = np.load(path)
    return img


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image

    Args:
        img: Image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    #function needed to flip the horizontal axis of a numpy array img
    mirrored_img = np.flip(img, axis=1)
    return mirrored_img


def display_images(img1, img2):
    """ Display the normal and the mirrored image in one plot

    Args:
        img1: First image to display
        img2: Second image to display
    """

    #displays the first image on one part of the plt
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('starting image')
    plt.imshow(img1)
    plt.axis('off')

    #displays the second image on the other part of the plt
    plt.subplot(1,2,2)
    plt.title('altered image')
    plt.imshow(img2)
    plt.axis('off')

    plt.show()
