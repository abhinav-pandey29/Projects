-----------------------------------------------------------------------------------

Project Title
-----------------------------------------------------------------------------------
Count Areas for binary Grey-scale images

-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------

Description
-----------------------------------------------------------------------------------
"count-areas" takes a grey-scale image represented as a 2-dimensional array of unsigned bytes as input.
 
It returns an array of 256 unsigned int numbers, each of them being a count of areas coloured with the corresponding shade of grey.

-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------

Getting Started
-----------------------------------------------------------------------------------
Open Terminal in the Project directory. Run the program using the following syntax:

	count-areas <filename> --shape <height> <width>

Example - 
For a binary image-file "sample.bin" with image height and width 256, 256 respectively use command,

	count-areas sample.bin --shape 256 256

-----------------------------------------------------------------------------------

Requirements
-----------------------------------------------------------------------------------
The project uses 'argparse' (to take arguments from the terminal/command prompt), 'numpy' (to create 2D input and output arrays) and python internal libraries.

The numpy library has been included in the project directory.
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------

Acknowledgements
-----------------------------------------------------------------------------------
Article title: Flood-fill Algorithm Tutorials | Algorithms | HackerEarth
URL: https://www.hackerearth.com/practice/algorithms/graphs/flood-fill-algorithm/tutorial/

-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------

Author
-----------------------------------------------------------------------------------
Abhinav Pandey
E-mail : abhinav_pandey29@yahoo.com

-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------