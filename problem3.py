import numpy as np
import matplotlib.pyplot as plt
import math


################################################################
#            DO NOT EDIT THESE HELPER FUNCTIONS                #
################################################################

# Plot 2D points
def displaypoints2d(points):
    plt.figure()
    plt.plot(points[0, :], points[1, :], '.b')
    plt.xlabel('Screen X')
    plt.ylabel('Screen Y')


# Plot 3D points
def displaypoints3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0,:], points[1,:], points[2,:], 'b')
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")

################################################################


def gettranslation(v):
    """ Returns translation matrix T in homogeneous coordinates 
    for translation by v.

    Args:
        v: 3d translation vector

    Returns:
        Translation matrix in homogeneous coordinates
    """
    # Create an identity 4x4 matrix
    translation_matrix = np.identity(4)

    # Set the top 3x3 submatrix to the identity matrix
    translation_matrix[:3, :3] = np.identity(3)

    # Set the last column to the translation vector
    translation_matrix[:3, 3] = v

    return translation_matrix


def getyrotation(d):
    """ Returns rotation matrix Ry in homogeneous coordinates for 
    a rotation of d degrees around the y axis.

    Args:
        d: degrees of the rotation

    Returns:
        Rotation matrix
    """
    radians = math.radians(d)  # Convert degrees to radians
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)

    # Create the rotation matrix
    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ], dtype=float)

    return rotation_matrix


def getxrotation(d):
    """ Returns rotation matrix Rx in homogeneous coordinates for a 
    rotation of d degrees around the x axis.

    Args:
        d: degrees of the rotation

    Returns:
        Rotation matrix
    """
    radians = math.radians(d)  # Convert degrees to radians
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)

    # Create the rotation matrix
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [0, 0, 0, 1]
    ], dtype=float)

    return rotation_matrix
    

def getzrotation(d):
    """ Returns rotation matrix Rz in homogeneous coordinates for a 
    rotation of d degrees around the z axis.

    Args:
        d: degrees of the rotation

    Returns:
        Rotation matrix
    """
    radians = math.radians(d)  # Convert degrees to radians
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)

    # Create the rotation matrix
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)

    return rotation_matrix


def getcentralprojection(principal, focal):
    """ Returns the (3 x 4) matrix L that projects homogeneous camera 
    coordinates on homogeneous image coordinates depending on the 
    principal point and focal length.

    Args:
        principal: the principal point, 2d vector
        focal: focal length

    Returns:
        Central projection matrix
    """
    # Create the central projection matrix
    L = np.array([
        [focal, 0, principal[0], 0],
        [0, focal, principal[1], 0],
        [0, 0, 1, 0]
    ], dtype=float)

    return L
    

def getfullprojection(T, Rx, Ry, Rz, L):
    """ Returns full projection matrix P and full extrinsic 
    transformation matrix M.

    Args:
        T: translation matrix
        Rx: rotation matrix for rotation around the x-axis
        Ry: rotation matrix for rotation around the y-axis
        Rz: rotation matrix for rotation around the z-axis
        L: central projection matrix

    Returns:
        P: projection matrix
        M: matrix that summarizes extrinsic transformations
    """
    # Combine the rotation matrices (Rx, Ry, Rz) into a single extrinsic transformation matrix M
    M = np.matmul(np.matmul(np.matmul(T, Rx), Ry), Rz)

    # Combine the extrinsic transformation matrix M with the central projection matrix L to get the full projection matrix P
    P = np.matmul(L, M)

    return P, M


def cart2hom(points):
    """ Transforms from cartesian to homogeneous coordinates.

    Args:
        points: a np array of points in cartesian coordinates

    Returns:
        A np array of points in homogeneous coordinates
    """
    # Check if the input array already has a fourth column of ones
    if points.shape[1] == 3:
        # If not, add a column of ones to the existing array
        homogeneous_points = np.column_stack((points, np.ones(points.shape[0])))
    else:
        homogeneous_points = points

    return homogeneous_points



def hom2cart(points):
    """ Transforms from homogeneous to cartesian coordinates.

    Args:
        points: a np array of points in homogenous coordinates

    Returns:
        A np array of points in cartesian coordinates
    """
    # Check if the input array has a fourth column (last element) that is not zero
    if np.all(points[:, -1] != 0):
        # Normalize the homogeneous coordinates by dividing each column by the last element
        cartesian_points = points[:, :-1] / points[:, -1][:, np.newaxis]
    else:
        raise ValueError("Homogeneous coordinates cannot be converted to Cartesian coordinates because of division by zero.")

    return cartesian_points


def loadpoints(path):
    """ Load 2d points from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array
    """
    try:
        data = np.load(path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {e}")



def loadz(path):
    """ Load z-coordinates from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array
    """
    try:
        data = np.load(path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {e}")


def invertprojection(L, P2d, z):
    """
    Invert just the projection L of cartesian image coordinates 
    P2d with z-coordinates z.

    Args:
        L: central projection matrix
        P2d: 2D image coordinates of the projected points
        z: z-components of the homogeneous image coordinates

    Returns:
        3D cartesian camera coordinates of the points
    """
    # Check if the input is valid
    if L.shape != (3, 4) or P2d.shape[0] != 2 or P2d.shape[1] != z.shape[0]:
        raise ValueError("Invalid input shapes")

    # Augment the 2D image coordinates P2d with z to create homogeneous image coordinates
    P3d_homogeneous = np.vstack((P2d, z))  # Stack P2d and z vertically

    # Invert the central projection matrix L using its pseudo-inverse
    L_inv = np.linalg.pinv(L)

    # Perform the inversion to get 3D Cartesian camera coordinates
    P3d_cartesian = np.dot(L_inv, P3d_homogeneous)

    return P3d_cartesian


def inverttransformation(M, P3d):
    """ Invert just the model transformation in homogeneous coordinates
    for the 3D points P3d in cartesian coordinates.

    Args:
        M: matrix summarizing the extrinsic transformations (4x4 matrix)
        P3d: 3D points in homogeneous coordinates with shape (4, num_points)

    Returns:
        3D points after the extrinsic transformations have been reverted
    """
    # Check if the input is valid
    if M.shape != (4, 4) or P3d.shape[0] != 4:
        raise ValueError("Invalid input shapes")

    # Invert the transformation for all 3D points individually
    num_points = P3d.shape[1]
    P3d_inverted = np.empty((3, num_points))

    for i in range(num_points):
        # Invert the transformation for the current point
        inverted_point = np.dot(M, P3d[:, i])

        # Extract the Cartesian coordinates (remove the last element)
        P3d_inverted[:, i] = inverted_point[:-1]

    return P3d_inverted


def projectpoints(P, X):
    """ Apply full projection matrix P to 3D points X in cartesian coordinates.

    Args:
        P: projection matrix (3x4 matrix)
        X: 3D points in Cartesian coordinates with shape (3, num_points)

    Returns:
        x: 2D points in Cartesian coordinates
    """
    # Check if the input is valid
    if P.shape != (3, 4) or X.shape[0] != 3:
        raise ValueError("Invalid input shapes")

    # Augment the 3D points with ones to create homogeneous coordinates
    num_points = X.shape[1]
    X_homogeneous = np.vstack((X, np.ones(num_points)))

    # Apply the projection matrix to the homogeneous 3D points
    x_homogeneous = np.dot(P, X_homogeneous)

    # Normalize the homogeneous 2D points to obtain Cartesian coordinates
    x = x_homogeneous[:2, :] / x_homogeneous[2, :]

    return x



def p3multiplechoice(): 
    '''
    Change the order of the transformations (translation and rotation).
    Check if they are commutative. Make a comment in your code.
    Return 0, 1 or 2:
    0: The transformations do not commute.
    1: Only rotations commute with each other.
    2: All transformations commute.
    '''

    return -1
