 #include <sys/stat.h>
#include <iostream>
using namespace std;

 int main(){
    struct stat results;

    if (stat("sample.bin", &results) == 0)
        // The size of the file in bytes is in
        cout<<results.st_size;
 }
