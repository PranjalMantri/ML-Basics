#include <stdio.h>
#include <stdlib.h>

const float learning_rate = 0.01; 
const int n_iters = 1000; 
float bias;
float *weights;

// Calculate dot product of two arrays
float dot_product(float matrix1[], float matrix2[], int length) {
    float result = 0;
    for (int i = 0; i < length; i++) {
        result += matrix1[i] * matrix2[i];
    }

    return result;
}

// Transpose a 2D array (We are converting the 2D array into a 1D array)
float *transpose(int n_samples, int n_features, float X[n_samples][n_features]) {
    float *X_T = (float *)malloc(n_features * n_samples * sizeof(float)); 
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X_T[j * n_samples + i] = X[i][j]; 
        }
    }
    return X_T;
}

// Calculate sum of elements in an array
float matrix_sum(float *array, int length) {
    float sum = 0;
    for (int i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum;
}

void fit(int n_samples, int n_features) {
   // Get training data X (Input variable) and Y (Output varaible)

    // Allocate memory for input data X
    float **X = (float **)malloc(n_samples * sizeof(float *));
    for (int i = 0; i < n_samples; i++) {
        X[i] = (float *)malloc(n_features * sizeof(float));
    }

    // Get input data for features
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            printf("Enter the value for sample %d and feature %d: ", i + 1, j + 1);
            scanf("%f", &X[i][j]);
        }
    }

    // Allocate memory for output data Y
    float *Y = (float *)malloc(n_samples * sizeof(float));

    // Get output values
    for (int i = 0; i < n_samples; i++) {
        printf("Enter the value for sample %d: ", i + 1);
        scanf("%f", &Y[i]);
    } 

   // Initialize weights and bias to 0
   
    // Allocate memory for weights
    weights = (float *)malloc(n_features * sizeof(float));
    for (int i = 0; i < n_features; i++) {
        weights[i] = 0;
    }

    bias = 0; // Initialize bias to 0

   // Perform gradient descent updates
    for (int iteration = 0; iteration < n_iters; iteration++) {
        float *y_pred = (float *)malloc(n_samples * sizeof(float));

        // Calculate y_pred and update weights
        for (int i = 0; i < n_samples; i++) {
            y_pred[i] = dot_product(X[i], weights, n_features) + bias;
        }

        // Calculate gradients
        float *error = (float *)malloc(n_samples * sizeof(float));
        for (int i = 0; i < n_samples; i++) {
            error[i] = y_pred[i] - Y[i];
        }

        // Updating weights ana bais
        for (int j = 0; j < n_features; j++) {
            float dw = 0;
            for (int i = 0; i < n_samples; i++) {
                dw += error[i] * X[i][j];
            }
            weights[j] -= (learning_rate / n_samples) * dw; 
        }

        float db = matrix_sum(error, n_samples); 
        bias -= (learning_rate / n_samples) * db; 

        free(error);
        free(y_pred); 
    }

    // Free allocated memory for input data
    for (int i = 0; i < n_samples; i++) {
        free(X[i]);
    }

    free(X);
    free(Y);
}

// Predict output for a new data point
float predict(float *new_data_point, int n_features) {
    float y_pred = dot_product(new_data_point, weights, n_features) + bias;
    return y_pred;
}

int main() {
    int n_samples, n_features;

    // Get training data size and features
    printf("Enter the number of samples in training data: ");
    scanf("%d", &n_samples);

    printf("Enter the number of features in training data: ");
    scanf("%d", &n_features);

   fit(n_samples, n_features);
   
   // Get new data point
    float *new_data_point = (float *)malloc(n_features * sizeof(float));
    printf("Enter the values for the new data point:\n");
    for (int i = 0; i < n_features; i++) {
        printf("Feature %d: ", i + 1);
        scanf("%f", &new_data_point[i]);
    }


    // predicting output for new data point
    float prediction = predict(new_data_point, n_features);
    printf("Predicted value for the new data point: %f\n", prediction);
   
    // Free remaining memory
    free(new_data_point);
    free(weights); 

    return 0;
}

// Example output:
// Enter the number of samples in training data: 5
// Enter the number of features in training data: 1
// Enter the value for sample 1 and feature 1: 1
// Enter the value for sample 2 and feature 1: 2
// Enter the value for sample 3 and feature 1: 3
// Enter the value for sample 4 and feature 1: 4
// Enter the value for sample 5 and feature 1: 5
// Enter the value for sample 1: 3
// Enter the value for sample 2: 7
// Enter the value for sample 3: 11
// Enter the value for sample 4: 15
// Enter the value for sample 5: 19
// Enter the values for the new data point:
// Feature 1: 6
// Predicted value for the new data point: 22.761349