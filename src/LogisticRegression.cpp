#include "LogisticRegression.h"
#include <cmath>
#include <iostream>

LogisticRegression::LogisticRegression(double lr, int iterations)
    : learning_rate(lr), num_iterations(iterations), b(0.0), w() {}

double LogisticRegression::sigmoid(double z) {
    // Sigmoid implementation
    return 1 / (1 + exp(-z));
}

double LogisticRegression::dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    // Dot product implementation
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

double LogisticRegression::compute_cost(const std::vector<std::vector<double>>& X, const std::vector<int>& y, 
                                        std::vector<double>& w, double& b) {
    // Cost function implementation
    int m = X.size(); // Number of examples
    double cost = 0.0;
    for (int i = 0; i < m; ++i) {
        double z = dot_product(X[i], w) + b;
        double f_wb = sigmoid(z);

        f_wb = std::max(std::min(f_wb, 1.0 - 1e-15), 1e-15);  // Avoid log(0) or log(1)

        cost += -y[i] * log(f_wb) - (1 - y[i]) * log(1 - f_wb);
    }
    return cost / m;
}

void LogisticRegression::compute_gradient(const std::vector<std::vector<double>>& X, 
                                                                             const std::vector<int>& y,
                                                                             std::vector<double>& w, double& b, double& dj_db,
                                                                             std::vector<double>& dj_dw) {
    // Gradient computation logic
    int m = X.size(); // Number of examples
    int n = X[0].size(); // Number of features

    dj_db = 0.0;
    std::fill(dj_dw.begin(), dj_dw.end(), 0.0);
    
    for (int i = 0; i < m; ++i) {
        double z = dot_product(X[i], w) + b;
        double f_wb = sigmoid(z);

        double dj_db_i = f_wb - y[i];
        dj_db += dj_db_i;

        for (int j = 0; j < n; ++j) {
            dj_dw[j] += dj_db_i * X[i][j];
        }
        
    }

    // Average the gradients
    for (int j = 0; j < n; ++j) {
        dj_dw[j] /= m;
    }
    dj_db /= m;
}

void LogisticRegression::gradient_descent(const std::vector<std::vector<double>>& X, const std::vector<int>& y, 
                                          std::vector<double>& w_in, double& b_in, int num_iterations, double learning_rate) {
    // Gradient descent logic
    int m = X.size();    // Number of examples
    int n = X[0].size(); // Number of features

    std::vector<double> J_history; 
    std::vector<std::vector<double> > w_history; 

    for (int i = 0; i < num_iterations; ++i) {
        std::vector<double> dj_dw(n, 0.0);
        double dj_db = 0.0;

        compute_gradient(X, y, w_in, b_in, dj_db, dj_dw);

        for (int j = 0; j < n; ++j) {
            w_in[j] -= learning_rate * dj_dw[j];
        }
        b_in -= learning_rate * dj_db;
        
        // Inside gradient descent
        //std::cout << "Iteration " << i << ": Bias (b) = " << b_in << std::endl;
        //std::cout << "Iteration " << i << ": Gradient of Bias (dj_db) = " << dj_db << std::endl;

    
        // Optional: Print cost every 1000 iterations
        if (i % 1000 == 0) {
            double cost = compute_cost(X, y, w_in, b_in);
            std::cout << "Iteration " << i << ": Cost = " << cost << std::endl;
        }
    }
}

std::vector<int> LogisticRegression::predict(const std::vector<std::vector<double>>& X) {
    // Prediction logic
    std::vector<int> predictions;
    for (const auto& x : X) {
        double z = dot_product(x, w) + b;

        predictions.push_back(sigmoid(z) >= 0.5 ? 1 : 0);
    }
    return predictions;
}

void LogisticRegression::normalize_features(std::vector<std::vector<double>>& X, const std::vector<double>& mean, const std::vector<double>& stddev) {
    // Normalization logic
    for (int j = 0; j < 4; ++j) {
        // Handle case where stddev is zero (zero variance)
        if (stddev[j] == 0.0) {
            std::cout << "Warning: Feature " << j << " has zero variance. Skipping normalization." << std::endl;
            continue; // Skip normalization for this feature
        }

        // Normalize the feature using provided mean and stddev
        for (auto& row : X) {
            row[j] = (row[j] - mean[j]) / stddev[j];
        }
    }
}

std::tuple<std::vector<double>, std::vector<double>> LogisticRegression::calc_mean_std(std::vector<std::vector<double>>& X, 
                                                                                       std::vector<double>& mean, std::vector<double>& stddev) {
    // Mean and standard deviation calculation logic
    int num_features = X[0].size();

    // Step 1: Calculate the mean for each feature (column)
    for (const auto& row : X) {
        for (int j = 0; j < num_features; ++j) {
            mean[j] += row[j];
        }
    }
    for (double& m : mean) {
        m /= X.size();  // Mean is the sum of all values divided by the number of samples
    }

    // Step 2: Calculate the standard deviation for each feature (column)
    for (const auto& row : X) {
        for (int j = 0; j < num_features; ++j) {
            stddev[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
        }
    }
    for (double& s : stddev) {
        s = sqrt(s / (X.size() - 1));  // Sample standard deviation (using N-1)
    }

    return make_tuple(mean, stddev);
}

