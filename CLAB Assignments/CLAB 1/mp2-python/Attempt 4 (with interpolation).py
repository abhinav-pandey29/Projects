# Code from Saurabh Gupta
from tqdm import tqdm
import os, sys, numpy as np, cv2

sys.path.insert(0, 'pybsds')
from scipy import signal
from skimage.util import img_as_float
from skimage.io import imread
from pybsds.bsds_dataset import BSDSDataset
from pybsds import evaluate_boundaries
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

GT_DIR = os.path.join('contour-data', 'groundTruth')
IMAGE_DIR = os.path.join('contour-data', 'images')
N_THRESHOLDS = 99


def get_imlist(name):
    imlist = np.loadtxt('contour-data/{}.imlist'.format(name))
    return imlist.astype(np.int)


def compute_edges_dxdy(I):
    """Returns the norm of dx and dy as the edge response function."""     
    I = I.astype(np.float32) / 255.
    I = cv2.GaussianBlur(I, (7,7), 10)
    
    scharr = np.array([[3, 0, -3],
                      [10, 0, -10],
                      [3, 0, -3]])
    
    dx = signal.convolve2d(I, scharr, mode='same', boundary='symm')
    dy = signal.convolve2d(I, scharr.T, mode='same', boundary='symm')
    
    grad_intensity = np.hypot(dx, dy)    
    grad_intensity = grad_intensity / grad_intensity.max() * 255
    
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
    
    #mag = np.sqrt(dx ** 2 + dy ** 2)
    mag = Z / np.max(Z)
    mag = mag * 255.
    
    # edge_labels = double_threshold(Z)
    # Z = prominent_edge_tracking(Z, edge_labels) 
    
    mag = np.clip(mag, 0, 255)  
    
    mag = mag.astype(np.uint8)
    
    return mag       

def double_threshold(im, low_thresh=0.4, high_thresh=0.60):
    
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

def non_max_suppression(img, D):
    
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
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

            except IndexError as e:
                pass
    
    return Z


def detect_edges(imlist, fn, out_dir):
    for imname in tqdm(imlist):
        I = cv2.imread(os.path.join(IMAGE_DIR, str(imname) + '.jpg'))
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        mag = fn(gray)
        out_file_name = os.path.join(out_dir, str(imname) + '.png')
        cv2.imwrite(out_file_name, mag)


def load_gt_boundaries(imname):
    gt_path = os.path.join(GT_DIR, '{}.mat'.format(imname))
    return BSDSDataset.load_boundaries(gt_path)


def load_pred(output_dir, imname):
    pred_path = os.path.join(output_dir, '{}.png'.format(imname))
    return img_as_float(imread(pred_path))


def display_results(ax, f, im_results, threshold_results, overall_result):
    out_keys = ['threshold', 'f1', 'best_f1', 'area_pr']
    out_name = ['threshold', 'overall max F1 score', 'average max F1 score',
                'area_pr']
    for k, n in zip(out_keys, out_name):
        print('{:>20s}: {:<10.6f}'.format(n, getattr(overall_result, k)))
        f.write('{:>20s}: {:<10.6f}\n'.format(n, getattr(overall_result, k)))
    res = np.array(threshold_results)
    recall = res[:, 1]
    precision = res[recall > 0.01, 2]
    recall = recall[recall > 0.01]
    label_str = '{:0.2f}, {:0.2f}, {:0.2f}'.format(
        overall_result.f1, overall_result.best_f1, overall_result.area_pr)
    # Sometimes the PR plot may look funny, such as the plot curving back, i.e,
    # getting a lower recall value as you lower the threshold. This is because of
    # the lack on non-maximum suppression. The benchmarking code does some
    # contour thinning by itself. Unfortunately this contour thinning is not very
    # good. Without having done non-maximum suppression, as you lower the
    # threshold, the contours become thicker and thicker and we lose the
    # information about the precise location of the contour. Thus, a thined
    # contour that corresponded to a ground truth boundary at a higher threshold
    # can end up far away from the ground truth boundary at a lower threshold.
    # This leads to a drop in recall as we decrease the threshold.
    ax.plot(recall, precision, 'r', lw=2, label=label_str)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')


if __name__ == '__main__':
    imset = 'val'
    imlist = get_imlist(imset)
    output_dir = 'contour-output/demo';
    fn = compute_edges_dxdy;
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Running detector:')
    detect_edges(imlist, fn, output_dir)

    _load_pred = lambda x: load_pred(output_dir, x)
    print('Evaluating:')
    sample_results, threshold_results, overall_result = \
        evaluate_boundaries.pr_evaluation(N_THRESHOLDS, imlist, load_gt_boundaries,
                                          _load_pred, fast=True, progress=tqdm)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    file_name = os.path.join(output_dir + '_out.txt')
    with open(file_name, 'wt') as f:
        display_results(ax, f, sample_results, threshold_results, overall_result)
    fig.savefig(os.path.join(output_dir + '_pr.pdf'), bbox_inches='tight')
