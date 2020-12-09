import numpy as np
import sys

# Filename and shape
filename = input('Enter file name : ')
x_dim = input('Enter x dimension : ')
y_dim = input('Enter y dimension : ')

shape = [int(x_dim), int(y_dim)]

# Initialize output array
output_array = [0] * 256


# Convert input image file to 2D array
def grey_to_array(file_name, _shape):
    in_array = np.fromfile(file_name, dtype='uint8')
    in_array = in_array.reshape(shape)

    return in_array


# Initialize input array
input_array = grey_to_array(filename, shape)


def flood_fill(x, y, old):

    to_fill = set()
    to_fill.add((x, y))
    while len(to_fill) != 0:
        (x, y) = to_fill.pop()

        if(x < shape[0] and y < shape[1]) and (input_array[x][y] == old):
            input_array[x][y] = True
            to_fill.add((x+1, y))
            to_fill.add((x, y-1))
            to_fill.add((x-1, y))
            to_fill.add((x, y+1))


def get_shade_index(shade):
    for row in range(256):
        for col in range(256):
            if input_array[row][col] == shade:
                return row, col
    return None, None


def is_complete():
    for row in input_array:
        for x in row:
            if x == False:
                return False
    return True


u = np.unique(input_array)


while not(is_complete()):
    for shade in u:
        row, col = get_shade_index(shade)
        if row != None:
            flood_fill(row, col, shade)
            output_array[shade] = output_array[shade] + 1


print(output_array)
print('Black: ', output_array[0])
print('Grey: ', output_array[200])
print('White: ', output_array[255])
