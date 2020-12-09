import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import signal

def detect(I=cv2.imread('contour-data/images/3096.jpg')):
    
  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)     
  I = I.astype(np.float32) / 255.
  I = cv2.GaussianBlur(I, (3,3), 1)
  
  scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                    [-10+0j, 0+ 0j, +10 +0j],
                    [ -3+3j, 0+10j,  +3 +3j]])
  
  # mag = np.absolute(signal.convolve2d(I, scharr, boundary='symm', mode='same'))

  dx = np.absolute(signal.convolve2d(I, scharr, mode='same'))
  dy = np.absolute(signal.convolve2d(I, scharr.T, mode='same'))
  
  G = np.hypot(dx, dy)
  G = G / G.max() * 255
  
  theta = np.arctan2(dx, dy)
  
  mag = non_max_suppression(G, theta)
  
  # mag = np.sqrt(dx ** 2 + dy ** 2)
  mag = mag / np.max(mag)
  mag = mag * 255.
  mag = np.clip(mag, 0, 255)
  mag = mag.astype(np.uint8)
    
  return mag

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            # try:
             q = 255
             r = 255
             
            #angle 0
             if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                 q = img[i, j+1]
                 r = img[i, j-1]
             #angle 45
             elif (22.5 <= angle[i,j] < 67.5):
                 q = img[i+1, j-1]
                 r = img[i-1, j+1]
             #angle 90
             elif (67.5 <= angle[i,j] < 112.5):
                 q = img[i+1, j]
                 r = img[i-1, j]
             #angle 135
             elif (112.5 <= angle[i,j] < 157.5):
                 q = img[i-1, j-1]
                 r = img[i+1, j+1]
    
             if (img[i,j] >= q) and (img[i,j] >= r):
                 Z[i,j] = img[i,j]
             else:
                 Z[i,j] = 0

            # except IndexError as e:
                # pass
    
    return Z


res = detect()
plt.imshow(res, cmap='gray')
plt.show()