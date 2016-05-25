#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
double activationFunction(double x)	{return 1.0/(1 + exp(-x));}
double dActivationFunction(double x)	{return activationFunction(x)*(1 - activationFunction(x));}
int main(int argc, char *argv[])
{
	if(argc!=13)
	{
		printf("Usage: ./execute samples features hiddenUnits outputDims learningRate batchSize epochs dataName labelName\n");
		printf("Example: ./execute 60000 10000 784 700 10 0.1 100 500 mnist_data_train.txt mnist_label_train.txt mnist_data_test mnist_label_test\n");
		exit(1);
	}
	int n = atoi(argv[1]);//60000
	int nt = atoi(argv[2]);//10000
	int d = atoi(argv[3]);//784
	int numberOfHiddenUnits = atoi(argv[4]);//700
	int outputDimensions = atoi(argv[5]);//10
	double learningRate = atof(argv[6]);//0.1
	int batchSize = atoi(argv[7]);//100
	int epochs = atoi(argv[8]);//500
	char * data_name = argv[9];
	char * label_name = argv[10]; 
	char * test_data_name = argv[11]; 
	char * test_label_name = argv[12];
	printf("n: %d, d: %d, numberOfHiddenUnits: %d, outputDimensions: %d\n", n, d, numberOfHiddenUnits, outputDimensions);
	printf("learningRate: %lf, batchSize: %d, epochs: %d\n", learningRate, batchSize, epochs);
	printf("data name: %s, label name: %s\n", data_name, label_name);
	double * data = (double *)malloc(n*d * sizeof(double));
	int * label = (int *)malloc(n * sizeof(double)); 
	int * labels = (int *)calloc(n*outputDimensions, sizeof(double)); 
	FILE * data_input = fopen(data_name, "r");
	FILE * label_input = fopen(label_name, "r");
	int i, j;
	for(i=0;i<n;i++)
	{
		fscanf(data_input, "%lf", data+i*d);
		for(j=1;j<d;j++) fscanf(data_input, ",%lf", data+i*d+j);
		fscanf(label_input, "%d", label+i);
	}
	fclose(data_input);
	fclose(label_input);
	//FILE * data_verify = fopen("data.verify", "w");
	//FILE * label_verify = fopen("label.verify", "w");
	//for(i=0;i<n;i++)
	//{
	//	fprintf(data_verify, "%lf", data[i*d+j]);
	//	for(j=1;j<d;j++) fprintf(data_verify, ",%lf", data[i*d+j]);
	//	fprintf(data_verify, "\n");
	//	fprintf(label_verify, "%d\n", label[i]);
	//}
	//return 0;
	for(i=0;i<n;i++)	labels[i*outputDimensions+label[i]]=1;
	srand(time(NULL));
	double * hiddenWeights = (double *)malloc(d * numberOfHiddenUnits * sizeof(double));
	//for(i=0;i<d*numberOfHiddenUnits;i++)	hiddenWeights[i] = rand()/d;
	for(i=0;i<d*numberOfHiddenUnits;i++)	hiddenWeights[i] = (double)rand()/(double)RAND_MAX/d;
	/*FILE * hw = fopen("hiddenWeights.txt","r");
	for(i=0;i<d;i++)
	{
		fscanf(hw, "%lf", hiddenWeights+i*numberOfHiddenUnits);
		for(j=1;j<numberOfHiddenUnits;j++)
		{
			fscanf(hw, ",%lf", hiddenWeights+i*numberOfHiddenUnits+j);
		}
	}
	fclose(hw);*/
	double * outputWeights = (double *)malloc(numberOfHiddenUnits * outputDimensions * sizeof(double));
	//for(i=0;i<numberOfHiddenUnits*outputDimensions;i++)	outputWeights[i] = rand()/numberOfHiddenUnits;
	for(i=0;i<numberOfHiddenUnits*outputDimensions;i++)	outputWeights[i] = (double)rand()/(double)RAND_MAX/numberOfHiddenUnits;
	/*FILE * ow = fopen("outputWeights.txt","r");
	for(i=0;i<numberOfHiddenUnits;i++)
	{
		fscanf(ow, "%lf", outputWeights+i*outputDimensions);
		for(j=1;j<outputDimensions;j++)
		{
			fscanf(ow, ",%lf", outputWeights+i*outputDimensions+j);
		}
	}
	fclose(ow);*/
	int * nbatch = (int *) calloc(batchSize, sizeof(int));
	double * inputVector = (double *)malloc(d * sizeof(double));
	double * hiddenActualInput = (double *)malloc(numberOfHiddenUnits * sizeof(double));
	double * hiddenOutputVector = (double *)malloc(numberOfHiddenUnits * sizeof(double));
	double * outputVector = (double *)malloc(outputDimensions * sizeof(double));
	double * targetVector = (double *)malloc(outputDimensions * sizeof(double));
	double * outputDelta = (double *)malloc(outputDimensions * sizeof(double));
	double * hiddenDelta = (double *)malloc(numberOfHiddenUnits * sizeof(double));
	double * error = (double *)malloc(outputDimensions * sizeof(double));
	int t, k;
	for(t=0;t<epochs;t++)
	{
		for(k=0;k<batchSize;k++)
		{
			int nb = nbatch[k] = (int)(rand() % n);
			int start = nb * d;
			for(j=0;j<d;j++)	inputVector[j] = data[start+j];
			for(i=0;i<numberOfHiddenUnits;i++)
			{
				double temp = 0.0;
				start = i * d;
				for(j=0;j<d;j++)	temp += hiddenWeights[start+j] * inputVector[j];
				hiddenActualInput[i] = temp; 
				hiddenOutputVector[i] = activationFunction(temp); 
			}
			int l = nb * outputDimensions;
			for(i=0;i<outputDimensions;i++)
			{
				double temp = 0.0;
				start = i * numberOfHiddenUnits;
				for(j=0;j<numberOfHiddenUnits;j++)	temp += outputWeights[start+j] * hiddenOutputVector[j];
				outputVector[i] = activationFunction(temp); 
				targetVector[i] = labels[l+i];
				error[i] = outputVector[i] - targetVector[i];
				outputDelta[i] = dActivationFunction(temp) * error[i];
			}
			for(j=0;j<numberOfHiddenUnits;j++)	hiddenDelta[j] = 0.0;
			for(i=0;i<outputDimensions;i++)
			{
				start = i * numberOfHiddenUnits;
				for(j=0;j<numberOfHiddenUnits;j++)	hiddenDelta[j] += outputWeights[start+j] * outputDelta[i];
			}
			for(j=0;j<numberOfHiddenUnits;j++)	hiddenDelta[j] = dActivationFunction(hiddenActualInput[j]) * hiddenDelta[j];
			for(i=0;i<outputDimensions;i++)
			{
				start = i * numberOfHiddenUnits;
				for(j=0;j<numberOfHiddenUnits;j++)
				{
					double temp = outputDelta[i] * hiddenOutputVector[j];
					outputWeights[start+j] -= learningRate * temp; 
				}
			}
			for(i=0;i<numberOfHiddenUnits;i++)
			{
				start = i * d;
				for(j=0;j<d;j++)
				{
					double temp = hiddenDelta[i] * inputVector[j];
					hiddenWeights[start+j] -= learningRate * temp;
				}
			}
		}
		double msd = 0.0;
		for(k=0;k<batchSize;k++)
                {
                        int nb = nbatch[k];
                        int start = nb * d;
                        for(j=0;j<d;j++)        inputVector[j] = data[start+j];
			start = nb * numberOfHiddenUnits;
			int l = nb * outputDimensions;
			for(i=0;i<numberOfHiddenUnits;i++)
                        {
                                double temp = 0.0;
                                start = i * d;
                                for(j=0;j<d;j++)        temp += hiddenWeights[start+j] * inputVector[j];
                                hiddenOutputVector[i] = activationFunction(temp);
                        }
			double err = 0.0;
			for(i=0;i<outputDimensions;i++)
                        {
                                double temp = 0.0;
                                start = i * numberOfHiddenUnits;
                                for(j=0;j<numberOfHiddenUnits;j++)      temp += outputWeights[start+j] * hiddenOutputVector[j];
                                outputVector[i] = activationFunction(temp);
                                targetVector[i] = labels[l+i];
                                error[i] = outputVector[i] - targetVector[i];
				err += error[i]*error[i];
                        }
			err = sqrt(err);
			msd += err;
		}
		msd = msd/batchSize;
		printf("****** Error of Step %d: %lf ******\n", t, msd);
	}
	double * test_data = (double *)malloc(nt * d * sizeof(double));
	int * test_label = (int *)malloc(nt * sizeof(double)); 
	FILE * test_data_input = fopen(test_data_name, "r");
	FILE * test_label_input = fopen(test_label_name, "r");
	printf("****** test_data_name: %s, test_label_name: %s ******\n", test_data_name, test_label_name);
	for(i=0;i<nt;i++)
	{
		fscanf(test_data_input, "%lf", test_data+i*d);
		for(j=1;j<d;j++) fscanf(test_data_input, ",%lf", test_data+i*d+j);
		fscanf(test_label_input, "%d", test_label+i);
	}
	fclose(test_data_input);
	fclose(test_label_input);
	int correct = 0;
	for(int nb=0;nb<nt;nb++)
        {
                int start = nb * d;
                for(j=0;j<d;j++)        inputVector[j] = test_data[start+j];
                for(i=0;i<numberOfHiddenUnits;i++)
                {
                        double temp = 0.0;
                        start = i * d;
                        for(j=0;j<d;j++)        temp += hiddenWeights[start+j] * inputVector[j];
                        hiddenOutputVector[i] = activationFunction(temp);
                }
		int group = -1;
		double max = -1;
                for(i=0;i<outputDimensions;i++)
                {
                        double temp = 0.0;
                        start = i * numberOfHiddenUnits;
                        for(j=0;j<numberOfHiddenUnits;j++)      temp += outputWeights[start+j] * hiddenOutputVector[j];
                        outputVector[i] = activationFunction(temp);
			if(max<outputVector[i])
			{
				max = outputVector[i];
				group = i;
			}
                }
		if(group==test_label[nb])	correct++;
        }
	printf("The accuracy is: %lf (%d/%d)\n", (double)correct/(double)nt, correct, nt);
	return 0;
}
