#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <cmath>
// #include "LogisticRegression.h"

using namespace std;

struct DataPoint {
    // Numerical features
    double funding_total_usd;
    double funding_rounds;
    double age_first_funding_year;
    double age_last_funding_year;
    double age_first_milestone_year;
    double age_last_milestone_year;
    double relationships;
    double milestones;

    // Binary features (0 or 1 values)
    double is_software;
    double is_web;
    double is_mobile;
    double is_enterprise;
    double is_advertising;
    double is_gamesvideo;
    double is_ecommerce;
    double is_biotech;
    double is_consulting;

    // Target variable
    int is_successful;  // 1 if successful (active), 0 if failed
};

// Utility function to convert strings to integers (used for missing values)
int stringToInt(const string &str) {
    try {
        // stoi converts string to int
        return stoi(str);
    } catch (...) {
        return -1;  // Return -1 if the conversion fails (e.g., missing data)
    }
}

// Utility function to convert strings to integers (used for missing values)
int stringToDouble(const string &str) {
    try {
        // stoi converts string to int
        return stod(str);
    } catch (...) {
        return -1;  // Return -1 if the conversion fails (e.g., missing data)
    }
}

// Class to handle data loading and preprocessing
class Dataset {
public:
    vector<DataPoint> data_points;

    // Loading data from file
    bool loadFromCSV(const string &filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            return false;
        }

        string line;
        getline(file, line);

        while (getline(file, line)){
            stringstream ss(line); // Stream for parsing the row
        DataPoint dp;

        string temp;

        // Skip irrelevant columns or parse them as needed
        getline(ss, temp, ','); // Unnamed: 0
        getline(ss, temp, ','); // state_code
        getline(ss, temp, ','); // latitude
        getline(ss, temp, ','); // longitude
        getline(ss, temp, ','); // zip_code
        getline(ss, temp, ','); // id
        getline(ss, temp, ','); // city
        getline(ss, temp, ','); // Unnamed: 6
        getline(ss, temp, ','); // name
        getline(ss, temp, ','); // labels
        getline(ss, temp, ','); // founded_at
        getline(ss, temp, ','); // closed_at
        getline(ss, temp, ','); // first_funding_at
        getline(ss, temp, ','); // last_funding_at

        // Extract relevant numerical features
        getline(ss, temp, ','); dp.age_first_funding_year = stringToInt(temp);
        getline(ss, temp, ','); dp.age_last_funding_year = stringToInt(temp);
        getline(ss, temp, ','); dp.age_first_milestone_year = stringToInt(temp);
        getline(ss, temp, ','); dp.age_last_milestone_year = stringToInt(temp);
        getline(ss, temp, ','); dp.relationships = stringToInt(temp);
        getline(ss, temp, ','); dp.funding_rounds = stringToInt(temp);
        getline(ss, temp, ','); dp.funding_total_usd = stringToInt(temp);
        getline(ss, temp, ','); dp.milestones = stringToInt(temp);

        // Extract binary features (e.g., is_CA, is_software, etc.)
        getline(ss, temp, ','); dp.is_software = stringToInt(temp);
        getline(ss, temp, ','); dp.is_web = stringToInt(temp);
        getline(ss, temp, ','); dp.is_mobile = stringToInt(temp);
        getline(ss, temp, ','); dp.is_enterprise = stringToInt(temp);
        getline(ss, temp, ','); dp.is_advertising = stringToInt(temp);
        getline(ss, temp, ','); dp.is_gamesvideo = stringToInt(temp);
        getline(ss, temp, ','); dp.is_ecommerce = stringToInt(temp);
        getline(ss, temp, ','); dp.is_biotech = stringToInt(temp);
        getline(ss, temp, ','); dp.is_consulting = stringToInt(temp);

        // Extract the target variable
        getline(ss, temp, ','); // object_id
        getline(ss, temp, ','); // has_VC
        getline(ss, temp, ','); // has_angel
        getline(ss, temp, ','); // has_roundA
        getline(ss, temp, ','); // has_roundB
        getline(ss, temp, ','); // has_roundC
        getline(ss, temp, ','); // has_roundD
        getline(ss, temp, ','); // avg_participants
        getline(ss, temp, ','); // is_top500
        getline(ss, temp, ','); dp.is_successful = stringToInt(temp);

        data_points.push_back(dp);

        }

        file.close();
        return true;
    }

    // Printing all data
    void displayData() const {
        int success_count = 0;
        int failure_count = 0;

        for (const auto &dp : data_points) {

            if (dp.is_successful) success_count++;
            else failure_count++;

            cout << "Success: " << success_count << ", Failure: " << failure_count << endl;
                cout << ", Funding: $" << dp.funding_total_usd
                        << ", Rounds: " << dp.funding_rounds
                        << ", Age at First Funding: " << dp.age_first_funding_year
                        << ", Relationships: " << dp.relationships
                        << ", Software: " << dp.is_software
                        << ", Web: " << dp.is_web
                        << ", Success: " << (dp.is_successful ? "Yes" : "No")
                        << endl;
                
            }
        cout << "Success: " << success_count << ", Failure: " << failure_count << endl;
    }

    // Split data into testing and training
    void splitData(vector<DataPoint>& trainingData, vector<DataPoint>& testingData){
        double ratio = 0.8;
        int splitIndex = data_points.size() * ratio;

        random_device rd;         // Random device for seeding
        default_random_engine rng(rd()); 

        shuffle(data_points.begin(), data_points.end(), rng);

        trainingData.reserve(splitIndex);                     // Allocate space for training data
        testingData.reserve(data_points.size() - splitIndex);

        copy(data_points.begin(), data_points.begin() + splitIndex, back_inserter(trainingData));
        copy(data_points.begin() + splitIndex + 1, data_points.end(), back_inserter(testingData));
    }


};


class LogisticRegression {
public:  

    vector<double> w; // Weights
    double b;              // Bias
    double learning_rate;  // Learning rate
    int num_iterations;

    LogisticRegression(double lr = 0.01, int iterations = 1000)
        : learning_rate(lr), num_iterations(iterations), b(0.0), w() {}

    double sigmoid(double z) {
        return 1 / (1 + exp(-z));
    }

    double dot_product(const vector<double>& a, const vector<double>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    double compute_cost(const vector<vector<double> >& X, const vector<double>& y, vector<double>& w, double& b) {
        int m = X.size(); // Number of examples
        double cost = 0.0;
        for (int i = 0; i < m; ++i) {
            double z = dot_product(X[i], w) + b;
            double f_wb = sigmoid(z);

            f_wb = max(min(f_wb, 1.0 - 1e-15), 1e-15);  // Avoid log(0) or log(1)

            cost += -y[i] * log(f_wb) - (1 - y[i]) * log(1 - f_wb);
        }
        return cost / m;
    }

    // Compute gradients
    tuple<vector<double>, double> compute_gradient(const vector<vector<double> >& X, const vector<double>& y, 
                          vector<double>& w, double& b) {
        int m = X.size(); // Number of examples
        int n = X[0].size(); // Number of features

        vector<double> dj_dw = vector<double>(n, 0.0);
        double dj_db = 0.0;

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
        //cout << "dj_dw: " << dj_dw[0] << ", dj_db: " << dj_db << endl;
        return make_tuple(dj_dw, dj_db);
    }

    vector<int> predict(const vector<vector<double> >& X) {

        vector<int> predictions;
        for (const auto& x : X) {
            double z = dot_product(x, w) + b;

            predictions.push_back(sigmoid(z) >= 0.5 ? 1 : 0);
        }
        return predictions;
    }

    tuple<vector<vector<double> >, vector<double>> gradient_descent(const vector<vector<double> >& X, const vector<double>& y, vector<double>& w_in, double& b_in, int num_iterations, double learning_rate) {
        int m = X.size();    // Number of examples
        int n = X[0].size(); // Number of features

        vector<double> J_history; 
        vector<vector<double> > w_history; 

        for (int i = 0; i < num_iterations; ++i) {
            vector<double> dj_dw(n, 0.0);
            double dj_db = 0.0;

            tie(dj_dw, dj_db) = compute_gradient(X, y, w_in, b_in);

            for (int j = 0; j < n; ++j) {
                w_in[j] -= learning_rate * dj_dw[j];
            }
            b_in -= learning_rate * dj_db;

        
            // Optional: Print cost every 100 iterations
            if (i % 1000 == 0) {
                double cost = compute_cost(X, y, w_in, b_in);
                cout << "Iteration " << i << ": Cost = " << cost << endl;
            }
        }

        return make_tuple(w_history, J_history);
    }

    // Normalize each feature column
    void normalize_features(vector<vector<double>>& X) {
        for (int j = 0; j < X[0].size(); ++j) {
            double mean = 0.0;
            double stddev = 0.0;
            
            // Compute mean of column
            for (const auto& row : X) {
                mean += row[j];
            }
            mean /= X.size();
            
            // Compute standard deviation of column
            for (const auto& row : X) {
                stddev += (row[j] - mean) * (row[j] - mean);
            }
            stddev = sqrt(stddev / X.size());
            
            // Normalize the column
            for (auto& row : X) {
                row[j] = (row[j] - mean) / stddev;
            }
        }
    }


};


int main() {
    Dataset dataset;
    LogisticRegression model(0.00001, 10000);

    // Vectors for training and testing data
    vector<DataPoint> trainingData;
    vector<DataPoint> testingData;


    vector<double> J_history; 
    vector<vector<double> > w_history;

    if (dataset.loadFromCSV("startup_data.csv")) {
        dataset.splitData(trainingData, testingData);
        cout << "Data loaded and preprocessed successfully!" << endl;

        // Step 2: Prepare feature matrix (X) and labels (y)
        vector<vector<double> > X_train;
        vector<double> y_train;

        // Prepare testing data
        vector<vector<double> > X_test;
        vector<int> y_test;

        for (const auto& dp : trainingData) {
            X_train.push_back({
                dp.funding_total_usd, dp.funding_rounds, dp.age_first_funding_year,
                dp.relationships, dp.is_software, dp.is_web, dp.is_mobile, dp.is_enterprise,
                dp.is_advertising, dp.is_gamesvideo, dp.is_ecommerce, dp.is_biotech,
                dp.is_consulting
            });
            y_train.push_back(dp.is_successful);
        }

        for (const auto& dp : testingData) {
            X_test.push_back({
                dp.funding_total_usd, dp.funding_rounds, dp.age_first_funding_year,
                dp.relationships, dp.is_software, dp.is_web, dp.is_mobile, dp.is_enterprise,
                dp.is_advertising, dp.is_gamesvideo, dp.is_ecommerce, dp.is_biotech,
                dp.is_consulting
            });
            y_test.push_back(dp.is_successful);
        }
    
        // Display dataset and feature matrix information
        //dataset.displayData();
        cout << "Total Data Points: " << dataset.data_points.size() << endl;
        cout << "Training Data: " << trainingData.size() << ", Testing Data: " << testingData.size() << endl;
        cout << "X_train: " << X_train.size() << ", y_train: " << y_train.size() << endl;
        
        model.w = vector<double>(X_train[0].size());
        for (double& weight : model.w) {
            weight = ((double)rand() / RAND_MAX) * 0.01; // Small random values
        }

        //cout << "Before Normalization:" << endl;
        //cout << "X_train size: " << X_train.size() << ", X_test size: " << X_test.size() << endl;

        model.normalize_features(X_train);
        model.normalize_features(X_test);

        //cout << "After Normalization:" << endl;
        //cout << "X_train size: " << X_train.size() << ", X_test size: " << X_test.size() << endl;

        // Train the model
        tie(w_history, J_history) = model.gradient_descent(X_train, y_train, model.w, model.b, model.num_iterations, model.learning_rate);

        vector<int> predictions_train = model.predict(X_train);

        int correct_train = 0;
        for (size_t i = 0; i < y_train.size(); ++i) {
            if (predictions_train[i] == y_train[i]) {
                ++correct_train;
            }
        }
        double accuracy_train = static_cast<double>(correct_train) / y_train.size();
        cout << "Training Accuracy: " << accuracy_train * 100 << "%" << endl;

        // Predict and evaluate
        vector<int> predictions = model.predict(X_test);

        int correct = 0;
        for (size_t i = 0; i < y_test.size(); ++i) {
            if (predictions[i] == y_test[i]) {
                ++correct;
            }
        }
        double accuracy = static_cast<double>(correct) / y_test.size();
        cout << "Model Accuracy: " << accuracy * 100 << "%" << endl;


    } else {
        cerr << "Failed to load data." << endl;
    }

    return 0;
}