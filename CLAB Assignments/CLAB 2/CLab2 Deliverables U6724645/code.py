"""
CLAB Task-1: Harris Corner Detector
Abhinav Pandey (u6724645):
"""

import numpy as np


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01
k = 0.06  # empirically determined constant in range [0.04,0.06] for Harris Corner Detection

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
import matplotlib.pyplot as plt

bw = plt.imread('Harris_1.png')
bw = bw = bw[:,:,0] * 0.299 + bw[:,:,1]*0.587 + bw[:,:,2] * 0.114 # Convert to grayscale
# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

## Creates a 13 x 13 Gaussian Kernel with standard deviation = sigma = 2
g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)


Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################
def harris_cornerness(Ixy, Ix2, Iy2, k):   
    
    # Step 1. Calculate the determinant value of the 2x2 moment matrix M = [[Ix2  Ixy],[Ixy  Iy2]]
    det_M = (Ix2 * Iy2) - np.power(Ixy, 2)
    
    # Step 2. Calculate the trace of M (sum of diagonal elements), Ix2 and Iy2
    trace_M = Ix2 + Iy2
    
    # Step 3. Compute the "cornerness" score : R = detM - k x traceM^2
    R = det_M - k * np.power(trace_M, 2)
    
    return R

R = harris_cornerness(Ixy, Ix2, Iy2, k)

######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################

def non_maximum_suppression(R, window_sizes, n=1):
    
    # Only 1 maximum point will be selected from non_overlapping window_sizes sized sections of R
    
    ## Window Sizes -
    ## Harris_1 = 4, 6
    ## Harris_2 = 4, 4
    ## Harris_3 = 7, 8
    ## Harris_4 = 5, 5
        
    rows, cols = R.shape  # Store shape of R
    window_r, window_c = window_sizes  # Specify window size 

    nms = np.zeros(R.shape)  # Create a zero matrix
    
    # Slice non-overlapping windows
    for i in np.arange(0, rows, window_r):
        for j in range(0, cols, window_c):

            y = R[i:i+window_r, j:j+window_c]
            
            # Select the maximum element from window and set the value of that position as 1 in nms
            for ele in y.flatten().argsort()[-n:]:
                nms[i + ele // window_r, j + ele % window_c] = 1
    
    # The product contains only maximum values of R in each tile
    nms_R = nms * R
    
    return nms_R

def thresholding(R, thresh):
    
    # Step 1. Set the threshold value = 0.01 x max(R)
    threshold = thresh * R.max()
    
    # Step 2. Fetch x and y coordinates of points with R > 0.01 x max(R)
    x, y = np.where(R > threshold)
    
    # Step 3. Combine coordinates into an N x 2 matrix
    corners = np.array([x,y]).T
    
    return corners

nms_R = non_maximum_suppression(R, (4,6), 1)
corners = thresholding(nms_R, thresh)


###########END TASK 1###############

"""
CLAB Task-2: K-Means Clustering and Color Image Segmentation
Abhinav Pandey (u6724645):
"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2

# Create data matrix from image
def image_to_vector(png_image, xy=True):  
#     xy = True ----> feature vectors with x-y coordinates
#     xy = False ----> feature vectors without x-y coordinates
    

    # Convert to L*a*b* representation
    lab_im = cv2.cvtColor(png_image, cv2.COLOR_RGB2Lab)
    
    # Store shape of image
    nrows, ncols = lab_im.shape[:2]
    
    # Create L*, a*, b* 1-D arrays corresponding to each of the image pixels
    L = lab_im[:,:, 0].flatten()
    a = lab_im[:,:, 1].flatten()
    b = lab_im[:,:, 2].flatten()
    
    if xy:
        # Returns a 1-D array corresponding to x-coordinates of pixels
        x = []
        for i in range(nrows):
            x += [i] * ncols
        x = np.array(x).flatten()

        # Returns a 1-D array corresponding to y-coordinates of pixels
        y = list(range(ncols)) * nrows
        y = np.array(y)
        
        # Features to include in output vector
        features = [L, a, b, x, y]
    
    else:    
        # Features to include in output vector
        features = [L,a,b]
        
    # Return output vector as numpy array
    image_vector = np.array(features).T
    
    return image_vector, nrows, ncols

# Random Initialisation strategy
def random_init(k, data):
    
    # The while loop ensures that all the selected datapoints are unique
    size=0  # Number of unique centroid ids
    while size != k:
        centroid_ids = np.random.randint(low=0, high=len(data), size=k)  # Generate k random indices
        size = len(set(centroid_ids))   # We want k unique indices
    
    # Centroids are selected using the random indices
    centroids = data[centroid_ids, :]

    return centroids

# Returns distance from a centroid
def dist_from_centroid(data, centroid):
    return ((data - centroid) ** 2).sum(axis=1)

# My KMeans++ Implementation
def my_kmeans_plus_init(k, data):
    
    # Step 1: Select a random datapoint from the data as a centroid
    centroids = data[np.random.randint(low=0, high=nrows),:]
    centroids

    for iteration in range(k-1):

        # Step 2 : Calculate distance of all datapoints to the closest centroid
        # a) Calculate distance from each centroid
        distance = np.array([dist_from_centroid(data, centroids[i]) for i in range(len(centroids))])
        # b) Store the smallest centroid distance for each datapoint
        distance = distance.min(axis=0)

        # Step 3 : Select the datapoint with maximum distance from initial centroid
        new_centroid = data[distance.argmax()]
        centroids = np.vstack((centroids, new_centroid))

    return centroids

# Assign index of nearest centroid as Label to all datapoints
def E_step(C, data):
    
    # k = number of cluster centres
    k = len(C)
    
    # Calculate distance matrix i.e. euclidean distance of each pixel from each centroid
    dist_matrix = ((data - C[:, None])**2).sum(axis=2).T
        
    # Determine closest cluster centers
    labels = dist_matrix.argmin(axis=1)
    
    return labels

# Readjust each centroid such that they are at the center of all datapoints belonging to that cluster
def M_step(C, L, data):
    
    # k = number of cluster centres
    k = len(C)
    
    # Update centroids for each of the k clusters (according to cluster labels, L)
    updated_centroids = [data[L==i].mean(axis=0) for i in range(k)]
    updated_centroids = np.array(updated_centroids)  
    
    return updated_centroids

# Implementation of K-means clustering
def my_kmeans(k, data, init_method, max_iterations=25):
    
    # Store the number of columns on the input data
    num_features = data.shape[1]
    
    # Initialize k centroids according to the desired initialization strategy
    if init_method == 'random':
        old_centroids = random_init(k, data)
    elif init_method == 'kmeans++':
        old_centroids = my_kmeans_plus_init(k, data)
    
    # Run the Expectation Maximisation Algorithm at most mox_iteration times
    for i in range(max_iterations):
        
        L = E_step(old_centroids, data)
 
        C = M_step(old_centroids, L, data)   
                
        if (old_centroids == C).all():  # If the centroids are unchanged, implies that the algorithm has converged
            break;
        
        old_centroids = C  # Book-keeping variable
        
    return C, L.reshape(nrows,ncols)  # Reshape Labels to the shape of the image

############## Implementation ##############

im = plt.imread("mandm.png")  # Read in RGB image 

# Generate 5D vector representation of the image - L*, a*, b*, x-coordinate, y-coordinate
kmeans_input, nrows, ncols = image_to_vector(im, True)
num_pixels = nrows * ncols

# Perform K-Means Clustering for k = 25
k = 25
C, L = my_kmeans(k, kmeans_input, init_method="random")   # Change init_method to "kmeans++" if desired

######## END TASK 2 ##############




"""
CLAB Task-3: Face Recognition using Eigenface
Abhinav Pandey (u6724645):
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

# List of filepaths for all training images
im_filepaths = glob.glob("trainingset/*")

# Store the image shape for reshaping purposes
image_shape = 231, 195

# Add all image matrices to a numpy array to create a big data matrix of images
training_images = np.array([plt.imread(image).flatten() for image in im_filepaths])

# Determine the average of all the training face images
avg_face = training_images.mean(axis=0)

# Normalise the training data by subtracting the mean image from each one
normalised_training_images = training_images - avg_face

# Compute the covariance matrix for the normalised images
cov_matrix = np.cov(normalised_training_images)

# Calculate the eigen-values and eigen-vectors of the covariance matrix
eig_values, eig_vectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors (wrt descending order of eigenvalues) for ordered Principal Components
desc_sorted_index = eig_values.argsort()[::-1]

eig_values = eig_values[desc_sorted_index]  # Sorted eigenvalues
eig_vectors = eig_vectors[:, desc_sorted_index] # Corresponding eigenvectors

# Assign k, where k is the number of Principal Components
k = 15

# Select the first k Principal Components (eigen-vectors)
k_components = eig_vectors[:k, :]

# Determine the eigen-faces using the first k Principal Components
eigen_faces = np.dot(k_components, normalised_training_images)

# Project eigen-faces onto k-dimensional subspace of eigen-vectors
model_weights = normalised_training_images @ eigen_faces.T



######## TESTING CODE#########

"""Returns a list 'test_preds' which contains 3 most similar images from training set to each test data"""

test_data_filepaths = glob.glob("testset/*")

# Vectorise test images
test_images = np.array([plt.imread(image).flatten() for image in test_data_filepaths])

# Normalise the test data by subtracting the average image from each one
normalised_test_images = test_images - avg_face

# Project test images onto the k Principal Component subspace
test_weights = normalised_test_images @ eigen_faces.T

# Calculate distance of all test images from each training image in the new subspace
dist_from_train_images = np.linalg.norm(test_weights[:,None] - model_weights, axis=2) 

# Determine the indexes of 3 "most similar" training images (those having least distances from each test image)

# a) Sorted indices of training images in order of decreasing similarity to test image
test_preds = dist_from_train_images.argsort(axis=1)  
# b) Select 3 most similar images (corresponding to first 3 indices)
test_preds = [test_preds[i][:3] for i in range(len(test_preds))]


########END TASK 3  ##############