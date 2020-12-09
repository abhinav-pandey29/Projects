bool is_complete(bool[][] _input_array){

for(int i=0; i < y_dim; i++){
    for(int j=0; j < x_dim; j++){

        if(_input_array[i][j] == false)
            return false;
    }
   return true;
 }
}

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

void flood_fill(int x, int y, int old, int[][] _input_array, int[] _shape){

    set <int, greater <int> > to_fill;
    to_fill.insert((x, y));
    while(to_fill.size != 0){

        (x, y) = to_fill.erase((x, y));

        if(((x < _shape[0]) & (y < _shape[1])) & (_input_array[x][y] == old){

            _input_array[x][y] = -1;
            to_fill.insert((x+1, y))
            to_fill.insert((x, y-1))
            to_fill.insert((x-1, y))
            to_fill.insert((x, y+1))

           }

    }

}
