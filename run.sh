rm execute;
mkdir data
cd data
wget http://www.cs.berkeley.edu/~youyang/data/mnist_data_train.txt
wget http://www.cs.berkeley.edu/~youyang/data/mnist_label_train.txt
wget http://www.cs.berkeley.edu/~youyang/data/mnist_data_test.txt
wget http://www.cs.berkeley.edu/~youyang/data/mnist_label_test.txt
cd ..
g++ -o execute bp.c;
./execute 60000 10000 784 700 10 0.1 100 500 ./data/mnist_data_train.txt ./data/mnist_label_train.txt ./data/mnist_data_test.txt ./data/mnist_label_test.txt
