import numpy as np
import sys
# import cv2
# import time

np.set_printoptions(threshold=sys.maxsize)
sample = np.fromfile('sample.bin', dtype='uint8')
sample = sample.reshape(256, 256)
a = [0] * 256


def flood_fill(x, y, old):

    toFill = set()
    toFill.add((x, y))
    while len(toFill) != 0:
        (x, y) = toFill.pop()

        # cv2.imwrite('sample2.png', sample)
        if(x < 256 and y < 256) and (sample[x][y] == old):
            sample[x][y] = True
            toFill.add((x+1, y))
            toFill.add((x-1, y))
            toFill.add((x, y+1))
            toFill.add((x, y-1))


u = np.unique(sample)


def get_shade_index(shade):
    for row in range(256):
        for col in range(256):
            if(sample[row][col] == shade):
                return (row, col)
    return (None, None)


def is_complete():
    for row in sample:
        for x in row:
            if(x != True):
                return False
    return True


while(not(is_complete())):
    for shade in u:
        row,col = get_shade_index(shade)
        if( row != None ):
            flood_fill(row, col, shade)
            a[shade] = a[shade] + 1


print(a)
print('Black: ', a[0])
print('Grey: ', a[200])
print('white: ', a[255])
# cv2.imwrite('sample2.png', sample)








