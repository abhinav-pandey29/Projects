# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#
I = Image.open('Left.jpg');

plt.imshow(I)
uv = plt.ginput(6) # Graphical user interface to get 6 points

#####################################################################
def calibrate(im, XYZ, uv):
    
    # Concatenate 1 to every real world coordinate : [X Y Z] --> [X Y Z 1]
    XYZ = np.concatenate((XYZ, np.ones((len(XYZ), 1))), axis=1)
    
    # Initialise lists to store the equation and result matrix
    A = []
    b = []
    
    # Repeat for each real world point (XYZ)
    # where each element of XYZ is of the form [X Y Z 1]
    for i, ele in enumerate(XYZ):
        
        u = uv[i, 0]  # Store x-coordinate of pixel corresponding to the real world point (xyz)
        v = uv[i, 1]  # Store y-coordinate of pixel corresponding to the real world point (xyz)
        
        eq1 = np.concatenate((ele, [0,0,0,0], -ele*u))[:-1]  # Store the first equation  
        eq2 = np.concatenate(([0,0,0,0], ele, -ele*v))[:-1]  # Store the second equation
        
        # Add the equations to the equation matrix, A
        A.append(eq1)
        A.append(eq2)
        
        # Add pixel coordinates to the solution matrix, b
        b.append(u)
        b.append(v)

    A, b = np.array(A), np.array(b)
    
    # Solve the equations in 2N x 11 matrix, A using solutions from 2N matrix, by using least squares method
    # to approximately determine the Camera Calibration matrix, C (of length 11)
    C, residuals = np.linalg.lstsq(A, b, rcond=None)[:2]
    
    # Display the error in satisfying camera calibration matrix constraints
    print("Error in satisfying the camera calibration matrix constraints = {}".format(residuals[0]))
    
    # The last element of C is 1
    C = np.concatenate((C, [1]))
    
    # Reshape matrix C to dimensions 3 x 4
    C = C.reshape((3,4))
    
    # Project XYZ onto the 2D image plane
    uv_preds = (C @ XYZ[:,:,None]).reshape((-1,3))
    uv_preds = uv_preds[:,:2] / uv_preds[:,-1, None] # Normalise the coordinates using the last element (u = uw/w)
    
    # Compute projection error, mean squared error
    mse = np.mean((uv_preds - uv)**2)
    
    # Load Vanishing points
    origin_and_vanishing_pts = np.load("Vanishing XYZ.npy")
    
    # Plotting Code
    plt.imshow(im)
    plt.scatter(uv[:,0], uv[:,1], s=10, marker='o', c='w')
    plt.scatter(uv_preds[:,0], uv_preds[:,1], s=10, marker='x', c='b')
    plt.title("Projection Error (MSE) = %.5f"%mse)
    
    # Visualize lines from the origin to the vanishing points in the X, Y and Z direction
    color = 'red' # Line color
    plt.plot(origin_and_vanishing_pts[:2,0], origin_and_vanishing_pts[:2,1], c=color)
    plt.plot(origin_and_vanishing_pts[1:3,0], origin_and_vanishing_pts[1:3,1], c=color)
    plt.plot(origin_and_vanishing_pts[[1,3],0], origin_and_vanishing_pts[[1,3],1], c=color)
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return C
'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% Abhinav Pandey (May 20, 2020)
'''

############################################################################
def homography(u2Trans, v2Trans, uBase, vBase):
    
    size = len(u2Trans)
    u2Trans, v2Trans = u2Trans.reshape((-1,1)), v2Trans.reshape((-1,1))
    
    trans_uv = np.concatenate((u2Trans, v2Trans, np.ones((size, 1))), axis=1)
    
    A = []

    for i, ele in enumerate(trans_uv):
        
        u = uBase[i]
        v = vBase[i]

        eq1 = np.concatenate(([0,0,0], -ele, ele * v))
        eq2 = np.concatenate(( -ele, [0,0,0], ele * u))

        A.append(eq1)
        A.append(eq2)


    A = np.array(A)

    # Decompose matrix A using SVD
    _, _, V = np.linalg.svd(A)
    
    # Reshape last column of V to get homography matrix 
    H = V.T[:,-1].reshape((3,3))  # Use the transpose of V since np.linalg.svd returns transpose of results
    
    # Divide matrix H with the last diagonal element
    H = H / H[2,2]    
    
    return H 

'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% Abhinav Pandey (May 20, 2020)
'''


############################################################################
def rq(A):
    # RQ factorisation

    [q,r] = np.linalg.qr(A.T)   # numpy has QR decomposition, here we can do it 
                                # with Q: orthonormal and R: upper triangle. Apply QR
                                # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R,Q

uv