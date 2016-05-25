rm execute;
g++ -o execute bp.c;
./execute 60000 10000 784 700 10 0.1 100 500 ./data/mnist_data_train.txt ./data/mnist_label_train.txt ./data/mnist_data_test.txt ./data/mnist_label_test.txt
