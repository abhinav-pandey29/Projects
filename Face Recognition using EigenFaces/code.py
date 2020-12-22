"""
Face Recognition using Eigenface
Author: Abhinav Pandey, Australian National University
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

# List of filepaths for all training images
im_filepaths = glob.glob("data/trainingset/*")

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

test_data_filepaths = glob.glob("data/testset/*")

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