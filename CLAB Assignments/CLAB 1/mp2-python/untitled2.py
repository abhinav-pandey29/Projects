import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal

def double_threshold(im, low_thresh=0.2, high_thresh=0.3):
    
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


im = plt.imread("contours.png")[:,:,0]
im = (im / im.max()) * 255

edge_labels = double_threshold(im)
new_im = prominent_edge_tracking(im, edge_labels)

plt.imshow(new_im, cmap='gray')
plt.show()