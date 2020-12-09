import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import signal

def double_threshold(im, low_thresh=0.25, high_thresh=0.35):
    
    low_thresh = low_thresh * im.max()
    high_thresh = high_thresh * im.max()
    
    labels = np.zeros((im.shape))
    labels[(im >= low_thresh) & (im < high_thresh)] = 1  # Weak Edges
    labels[im > high_thresh] = 2  # Strong Edges
    
    return labels

def prominent_edge_tracking(im, labels):
    
    M, N = im.shape
    for i in range(M-2):
        for j in range(N-2):
            
            im_window = im[i:i+3, j:j+3]
            window_labels = labels[i:i+3, j:j+3]
            
            centre = im[i+1, j+1]
            centre_label = labels[i+1, j+1]
            
            if centre_label == 2:
                window_labels[window_labels == 1] = 2
    
    new_im = im.copy()
    new_im[labels != 2] = 0
    
    return new_im

# Load the image
img = cv2.imread('contour-data/images/3096.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
I = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# Blurring for removing the noise 
I = cv2.GaussianBlur(I, (7,7), 10)

dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')

grad_intensity = np.hypot(dx, dy)

thresh = 0.15 * grad_intensity.max()
grad_intensity[grad_intensity < thresh] = 0

angle = np.arctan2(dy, dx)
    
Z = np.zeros((grad_intensity.shape))

for i in range(grad_intensity.shape[0] - 1):
    for j in range(grad_intensity.shape[1] - 1):
        
        try:
            theta = angle[i,j]  # Direction of gradient at current pixel
            theta_mod = theta % np.pi  # Positive angle
            
            q, r = 255,255
            
            if (0 <= theta_mod < np.pi/4):
                alpha = np.abs(np.tan(theta_mod))
                q = (alpha * grad_intensity[i + 1, j + 1]) + ((1 - alpha) * grad_intensity[i, j + 1])
                r = (alpha * grad_intensity[i - 1, j - 1]) + ((1 - alpha) * grad_intensity[i, j - 1]) 
            
            elif (np.pi/4 <= theta_mod < np.pi/2):
                alpha = np.abs(1./np.tan(theta_mod))
                q = (alpha * grad_intensity[i + 1, j + 1]) + ((1 - alpha) * grad_intensity[i + 1, j])
                r = (alpha * grad_intensity[i - 1, j - 1]) + ((1 - alpha) * grad_intensity[i - 1, j])
            
            elif (np.pi/2 <= theta_mod < (3*np.pi/4)):
                alpha = np.abs(1./np.tan(theta_mod))
                q = (alpha * grad_intensity[i + 1, j - 1]) + ((1 - alpha) * grad_intensity[i + 1, j])
                r = (alpha * grad_intensity[i - 1, j + 1]) + ((1 - alpha) * grad_intensity[i - 1, j])
            
            elif ((3*np.pi/4) <= theta_mod < np.pi):
                alpha = np.abs(np.tan(theta_mod))
                q = (alpha * grad_intensity[i + 1, j - 1]) + ((1 - alpha) * grad_intensity[i, j - 1])
                r = (alpha * grad_intensity[i - 1, j + 1]) + ((1 - alpha) * grad_intensity[i, j + 1])
    
            if (grad_intensity[i,j] >= q) and (grad_intensity[i,j] >= r):
                Z[i,j] = grad_intensity[i,j]
            else:
                Z[i,j] = 0
    
        except IndexError:
            pass 


edge_labels = double_threshold(Z)
Z = prominent_edge_tracking(Z, edge_labels)
plt.imshow(Z, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()


# # Apply the thresholding
# a = img_gray.max()  
# _, thresh = cv2.threshold(img_gray, a/2 +10, a,cv2.THRESH_BINARY_INV)
# plt.imshow(thresh, cmap = 'gray')

# # Find the contour of the figure 
# image, contours, hierarchy = cv2.findContours(
#                                    image = thresh, 
#                                    mode = cv2.RETR_EXTERNAL, 
#                                    method = cv2.CHAIN_APPROX_SIMPLE)

# # Sort the contours 
# contours = sorted(contours, key = cv2.contourArea, reverse = True)
# # Draw the contour 
# img_copy = img.copy()
# final = cv2.drawContours(img_copy, contours, contourIdx = -1, 
#                          color = (255, 0, 0), thickness = 2)
# plt.imshow(img_copy)

# # The first order of the contours
# c_0 = contours[0]
# # image moment
# M = cv2.moments(c_0)
# print(M.keys())

# # The area of contours 
# print("1st Contour Area : ", cv2.contourArea(contours[0])) 
# print("2nd Contour Area : ", cv2.contourArea(contours[1])) 
# print("3rd Contour Area : ", cv2.contourArea(contours[2]))

# # The arc length of contours 
# print(cv2.arcLength(contours[0], closed = True))      
# print(cv2.arcLength(contours[0], closed = False))     

# # The centroid point
# cx = int(M['m10'] / M['m00'])
# cy = int(M['m01'] / M['m00'])

# # The extreme points
# l_m = tuple(c_0[c_0[:, :, 0].argmin()][0])
# r_m = tuple(c_0[c_0[:, :, 0].argmax()][0])
# t_m = tuple(c_0[c_0[:, :, 1].argmin()][0])
# b_m = tuple(c_0[c_0[:, :, 1].argmax()][0])
# pst = [l_m, r_m, t_m, b_m]
# xcor = [p[0] for p in pst]
# ycor = [p[1] for p in pst]

# # Plot the points
# plt.figure(figsize = (10, 16))
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap = 'gray')
# plt.scatter([cx], [cy], c = 'b', s = 50)
# plt.subplot(1, 2, 2)
# plt.imshow(image, cmap = 'gray')
# plt.scatter(xcor, ycor, c = 'b', s = 50)