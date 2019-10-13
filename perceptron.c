#include <stdio.h>

#define learning_rate 1


/*
 * Function: step_function
 * ----------------------------
 *   threshold-based activation function
 *
 *   perceptron_output: dot product of input data and perceptron weights
 *
 *
 *   returns: 1 if the perceptron_output is bigger than 0, else 0
 */
int step_function(int perceptron_output) {
    return (perceptron_output < 0) ? 0 : 1;
}


/*
 * Function: perceptron_
 * ----------------------------
 *   supervised learning algorithm for binary classification
 *
 *   rows: rows of the dataset
 *   columns: columns of the dataset
 *   dataset: 2d dataset
 *   desired_output: desired classification value
 *   epochs: number of iterations over the dataset
 *
 *
 *   returns: 1 for success
 */
int perceptron_(int rows, int columns, int dataset[][2], int
    desired_output[4]) {

    // Variable Initialization
    int weights[columns] = {0}, bias = 0, actual_output = 0, error = 0;

    /*
     *  The classic Perceptron by F. Rosenblatt
     *  Uses the Perceptron Learning Rule for the optimization of the weights.
     *  W_new = W_old + learning_rate * error * Xij
     *
     *  A step/threshold function as the activation function.
     */
    for (int i = 0; i < rows; i++) {

        actual_output = 0;
        for (int j = 0; j < columns; j++) {
            actual_output = actual_output + weights[j] * dataset[i][j];
        }
        actual_output = step_function(actual_output + bias);

        error = desired_output[i] - actual_output;

        // Weights Optimization
        for (int j = 0; j < columns; j++) {
            weights[j] = weights[j] + learning_rate * error * dataset[i][j];
        }
        bias = bias + learning_rate * error * 1;
    }

    return 1;
}

/*
 * Function: perceptron_train
 * ----------------------------
 *  as perceptron_ with epoch implementation
 *
 */
int perceptron_train(int rows, int columns, int dataset[][2], int
    target_values[4], int epochs) {

    for (int i; i <= epochs; i++) {
        perceptron_(rows, columns, dataset, target_values);
    }

    return 1;
}

int main(void) {
    return 1;
}
