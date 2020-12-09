from GreyCnt import area_count
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(dest='filename', help="File Name")
parser.add_argument("-shape", "--shape", action='append', nargs=2, help="Shape")

args = parser.parse_args()

filename = args.filename
shape = args.shape
shape = shape.pop()

img_shape = []
img_shape.append(int(shape[1]))
img_shape.append(int(shape[0]))


# Output Array
count_array = area_count(filename, img_shape)

for ele in count_array:
    print(ele)
