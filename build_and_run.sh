mkdir -p build && cd build
cmake .. && make

export LD_LIBRARY_PATH=../lib/
./testseg /home/nreal/dataset/2022-0908-083512
