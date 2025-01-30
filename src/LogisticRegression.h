#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>
#include <tuple>

class LogisticRegression {
public:
    LogisticRegression(double lr = 0.01, int iterations = 1000);

    double sigmoid(double z);
    double dot_product(const std::vector<double>& a, const std::vector<double>& b);
    double compute_cost(const std::vector<std::vector<double>>& X, const std::vector<double>& y, 
                        std::vector<double>& w, double& b);
    std::tuple<std::vector<double>, double> compute_gradient(const std::vector<std::vector<double>>& X, 
                                                             const std::vector<double>& y,
                                                             std::vector<double>& w, double& b);
    void gradient_descent(const std::vector<std::vector<double>>& X, const std::vector<double>& y, 
                          std::vector<double>& w_in, double& b_in, int num_iterations, double learning_rate);
    std::vector<int> predict(const std::vector<std::vector<double>>& X);
    void normalize_features(std::vector<std::vector<double>>& X, const std::vector<double>& mean, const std::vector<double>& stddev);
    std::tuple<std::vector<double>, std::vector<double>> calc_mean_std(std::vector<std::vector<double>>& X, 
                                                                       std::vector<double>& mean, std::vector<double>& stddev);

public:
    std::vector<double> w; 
    double b;              
    double learning_rate;
    int num_iterations;
};

#endif
