#include <stdio.h>
#include <string.h>

#define array_length(array) (sizeof((array))/sizeof((array)[0]))
#define learning_rate 1

typedef struct weights{
    int bias;
    int w_length;
    int w[];
} model_weights;

typedef enum {
    activation_step_function
} activation_function_type;


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
model_weights perceptron_train(int rows, int columns, int dataset[][2], int
        desired_output[4], int epochs) {

    // Variable Initialization
    int weights[columns] = {0}, bias = 0, actual_output = 0, error = 0;

    /*
     *  The classic Perceptron by F. Rosenblatt
     *  Uses the Perceptron Learning Rule for the optimization of the weights.
     *  W_new = W_old + learning_rate * error * Xij
     *
     *  A step/threshold function as the activation function.
     */
    for (int repetitions; repetitions <= epochs; repetitions++) {
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
    }

    model_weights w;
    w.bias = bias;
    w.w_length = array_length(weights);
    // Arrays are not directly assignable
    memcpy(w.w, weights, sizeof(weights));


    return w;
}


int activation_function(int value, activation_function_type a_f) {

    switch (a_f) {
        case activation_step_function:
            return step_function(value);
        default:
            break;
    }

    return 1;
}


int perceptron_predict(int x[], model_weights w, activation_function_type
        a_f) {

    int actual_output = 0;

    for (int i = 0; i < w.w_length; i++) {
        actual_output = actual_output + x[i] * w.w[i];
    }
    actual_output = actual_output + w.bias;
    actual_output = activation_function(actual_output, a_f);

    return actual_output;
}

int main(void) {

    int X[][2] = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
    };

    int Y[] = {0, 0, 0, 1};

    model_weights test = perceptron_train(4, 2, X, Y, 100);
    printf("%d", test.bias);

    int x1[] = {1, 1};
    activation_function_type step_function = step_function;

//
//    int output = perceptron_predict(Y1, test, step_function);

    getchar();
    return 3;
}
