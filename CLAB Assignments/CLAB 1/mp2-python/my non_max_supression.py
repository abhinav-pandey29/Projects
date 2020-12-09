import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal

I = cv2.imread("contour-data/images/3096.jpg")
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) 

I = I.astype(np.float32) / 255.
I = cv2.GaussianBlur(I, (7,7), 10)

scharr = np.array([[3, 0, -3],
                  [10, 0, -10],
                  [3, 0, -3]])

dx = signal.convolve2d(I, scharr, mode='same', boundary='symm')
dy = signal.convolve2d(I, scharr.T, mode='same', boundary='symm')

grad_intensity = np.hypot(dx, dy)
grad_intensity = grad_intensity / grad_intensity.max() * 255

angle = np.arctan2(dx, dy)


M, N = grad_intensity.shape
Z = np.zeros((M,N))
# padded_grad_intensity[1:-1,1:-1] = grad_intensity

for i in range(grad_intensity.shape[0] - 1):
    for j in range(grad_intensity.shape[1] - 1):
        
        try:
            theta = angle[i,j]  # Direction of gradient at current pixel
            theta_mod = theta % np.pi  # Positive angle
            
            # q, r = 255,255
            
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

#mag = np.sqrt(dx ** 2 + dy ** 2)
mag = Z / np.max(Z)
mag = mag * 255.

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

edge_labels = double_threshold(mag)
mag = prominent_edge_tracking(mag, edge_labels)
mag = np.clip(mag, 0, 255)
# mag = mag.astype(np.uint8)

plt.imshow(mag, cmap='gray')
plt.show()
  
