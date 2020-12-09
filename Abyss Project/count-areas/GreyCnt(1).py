import numpy as np


# Convert input image file to 2D array
def grey_to_array(file_name, _shape):
    in_array = np.fromfile(file_name, dtype='uint8')
    in_array = in_array.reshape(_shape)

    return in_array


# Traversing 2D array in a Flood Fill manner
def flood_fill(x, y, old, _input_array, _shape):

    to_fill = set()
    to_fill.add((x, y))
    while len(to_fill) != 0:
        (x, y) = to_fill.pop()

        if(x < _shape[0] and y < _shape[1]) and (_input_array[x][y] == old):
            _input_array[x][y] = True
            to_fill.add((x+1, y))
            to_fill.add((x, y-1))
            to_fill.add((x-1, y))
            to_fill.add((x, y+1))


# Obtaining Index position of given Shade
def get_shade_index(shade, _input_array, _shape):
    for row in range(_shape[0]-1, -1, -1):
        for col in range(_shape[1]-1, -1, -1):
            if _input_array[row][col] == shade:
                return row, col
    return None, None


# Setting TRUE for elements that have been traversed
def is_completed(_input_array):
    for row in _input_array:
        for x in row:
            if x == False:
                return False
    return True


# Increment total count for each color segment
def area_count(_filename, _shape):

    # Initialising input and output array
    _input_array = grey_to_array(_filename, _shape)
    output_array = [0] * 256

    # Making a list of the unique colors in the image
    unique_ele = np.unique(_input_array)

    # Counting the number of colored areas for each unique color in the image
    while not(is_completed(_input_array)):
        for shade in unique_ele:
            row, col = get_shade_index(shade, _input_array, _shape)
            if row != None:
                flood_fill(row, col, shade, _input_array, _shape)
                output_array[shade] = output_array[shade] + 1

    return output_array
