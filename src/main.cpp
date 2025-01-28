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

// Utility function to convert strings to double (used for missing values)
int stringToDouble(const string &str) {
    try {
        // stoi converts string to double
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
    void splitData(vector<DataPoint>& allData, vector<DataPoint>& trainingData, vector<DataPoint>& devData, vector<DataPoint>& testingData){
        double trainRatio = 0.6;
        double devRatio = 0.2;

        int trainSplitIndex = data_points.size() * trainRatio;
        int devSplitIndex = trainSplitIndex + data_points.size() * devRatio;

        random_device rd;         // Random device for seeding
        default_random_engine rng(rd()); 

        shuffle(data_points.begin(), data_points.end(), rng);

        trainingData.reserve(trainSplitIndex);                     // Allocate space for training data
        devData.reserve(devSplitIndex - trainSplitIndex);
        testingData.reserve(data_points.size() - devSplitIndex);

        copy(data_points.begin(), data_points.end(), back_inserter(allData));
        copy(data_points.begin(), data_points.begin() + trainSplitIndex, back_inserter(trainingData));
        copy(data_points.begin() + trainSplitIndex, data_points.begin() + devSplitIndex, back_inserter(devData));
        copy(data_points.begin() + devSplitIndex, data_points.end(), back_inserter(testingData));
    }


};

// Logistic Regression class, including sigmoid, dot product, cost and gradient descent
class LogisticRegression {
public:  

    vector<double> w; // Weights
    double b;              // Bias
    double learning_rate;  // Learning rate
    int num_iterations;

    LogisticRegression(double lr = 0.01, int iterations = 1000)
        : learning_rate(lr), num_iterations(iterations), b(0.0), w() {}

    // Sigmoid function
    double sigmoid(double z) {
        return 1 / (1 + exp(-z));
    }

    // Dot product function
    double dot_product(const vector<double>& a, const vector<double>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    // Cost function
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

    // Predict function
    vector<int> predict(const vector<vector<double> >& X) {

        vector<int> predictions;
        for (const auto& x : X) {
            double z = dot_product(x, w) + b;

            predictions.push_back(sigmoid(z) >= 0.5 ? 1 : 0);
        }
        return predictions;
    }

    // Gradient descent function
    void gradient_descent(const vector<vector<double> >& X, const vector<double>& y, vector<double>& w_in, double& b_in, int num_iterations, double learning_rate) {
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
            
            // Inside gradient descent
            //cout << "Iteration " << i << ": Bias (b) = " << b_in << endl;
            //cout << "Iteration " << i << ": Gradient of Bias (dj_db) = " << dj_db << endl;

        
            // Optional: Print cost every 100 iterations
            if (i % 1000 == 0) {
                double cost = compute_cost(X, y, w_in, b_in);
                //cout << "Iteration " << i << ": Cost = " << cost << endl;
            }
        }
    }

    void normalize_features(vector<vector<double>>& X, const vector<double>& mean, const vector<double>& stddev) {
        // Loop through the first 4 features (columns)
        for (int j = 0; j < 4; ++j) {
            // Handle case where stddev is zero (zero variance)
            if (stddev[j] == 0.0) {
                cout << "Warning: Feature " << j << " has zero variance. Skipping normalization." << endl;
                continue; // Skip normalization for this feature
            }

            // Normalize the feature using provided mean and stddev
            for (auto& row : X) {
                row[j] = (row[j] - mean[j]) / stddev[j];
            }
        }
    }


   tuple<vector<double>, vector<double>> calc_mean_std(vector<vector<double>>& X_alldata, vector<double>& mean, vector<double>&stddev){
        int num_features = X_alldata[0].size();

        // Step 1: Calculate the mean for each feature (column)
        for (const auto& row : X_alldata) {
            for (int j = 0; j < num_features; ++j) {
                mean[j] += row[j];
            }
        }
        for (double& m : mean) {
            m /= X_alldata.size();  // Mean is the sum of all values divided by the number of samples
        }

        // Step 2: Calculate the standard deviation for each feature (column)
        for (const auto& row : X_alldata) {
            for (int j = 0; j < num_features; ++j) {
                stddev[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
            }
        }
        for (double& s : stddev) {
            s = sqrt(s / (X_alldata.size() - 1));  // Sample standard deviation (using N-1)
        }

        return make_tuple(mean, stddev);
        }

};

void predict_user_startup(LogisticRegression& model, vector<double>& mean, vector<double>&stddev) {
    vector<double> user_input;

    cout << "Enter details about your startup:" << endl;
    double funding_total_usd, funding_rounds, age_first_funding_year, relationships;
    double is_software, is_web, is_mobile, is_enterprise, is_advertising, is_gamesvideo, is_ecommerce, is_biotech, is_consulting;

    cout << "Funding Total (USD): ";
    cin >> funding_total_usd;
    cout << "Funding Rounds: ";
    cin >> funding_rounds;
    cout << "Age at First Funding Year: ";
    cin >> age_first_funding_year;
    cout << "Number of Relationships: ";
    cin >> relationships;

    cout << "Is your company Software-related? (1 for Yes, 0 for No): ";
    cin >> is_software;
    cout << "Is it Web-based? (1 for Yes, 0 for No): ";
    cin >> is_web;
    cout << "Is it Mobile-focused? (1 for Yes, 0 for No): ";
    cin >> is_mobile;
    cout << "Is it Enterprise-oriented? (1 for Yes, 0 for No): ";
    cin >> is_enterprise;
    cout << "Is it focused on Advertising? (1 for Yes, 0 for No): ";
    cin >> is_advertising;
    cout << "Is it in Games/Video? (1 for Yes, 0 for No): ";
    cin >> is_gamesvideo;
    cout << "Is it in E-commerce? (1 for Yes, 0 for No): ";
    cin >> is_ecommerce;
    cout << "Is it in Biotech? (1 for Yes, 0 for No): ";
    cin >> is_biotech;
    cout << "Is it a Consulting company? (1 for Yes, 0 for No): ";
    cin >> is_consulting;

    // Prepare user input vector for continuous and categorical features separately
    vector<double> continuous_input = {
        funding_total_usd, funding_rounds, age_first_funding_year, relationships
    };
    vector<double> categorical_input = {
        is_software, is_web, is_mobile, is_enterprise,
        is_advertising, is_gamesvideo, is_ecommerce, is_biotech, is_consulting
    };

    // Normalize continuous features only
    vector<vector<double>> continuous_input_vector = {continuous_input};
    model.normalize_features(continuous_input_vector, mean, stddev);

    // Combine normalized continuous features with original categorical features
    user_input.clear();
    user_input.insert(user_input.end(), continuous_input_vector[0].begin(), continuous_input_vector[0].end());
    user_input.insert(user_input.end(), categorical_input.begin(), categorical_input.end());

    cout << "Normalized user input: ";
    for (const auto& feature : user_input) {
        cout << feature << " ";  // This will print each normalized feature
    }
    cout << endl;

    // Predict success probability
    double probability = model.sigmoid(model.dot_product(user_input, model.w) + model.b);

    // Output the result
    cout << "Your startup's predicted success probability is: " << probability * 100 << "%" << endl;
    if (probability >= 0.5) {
        cout << "Prediction: Likely to Succeed!" << endl;
    } else {
        cout << "Prediction: Less likely to Succeed." << endl;
    }
}


int main() {
    Dataset dataset;
    LogisticRegression model(0.1, 100);

    // Vectors for training and testing data
    vector<DataPoint> allData;
    vector<DataPoint> trainingData;
    vector<DataPoint> devData;
    vector<DataPoint> testingData;

    if (dataset.loadFromCSV("startup_data.csv")) {
        dataset.displayData();
        dataset.splitData(allData, trainingData, testingData, devData);
        //cout << "Data loaded and preprocessed successfully!" << endl;

        vector<vector<double> > X_allData;
        // Step 2: Prepare feature matrix (X) and labels (y)
        vector<vector<double> > X_train;
        vector<double> y_train;

        vector<vector<double> > X_dev;
        vector<double> y_dev;

        // Prepare testing data
        vector<vector<double> > X_test;
        vector<int> y_test;

        for (const auto& dp : allData) {
            X_allData.push_back({
                dp.funding_total_usd, dp.funding_rounds, dp.age_first_funding_year,
                dp.relationships, dp.is_software, dp.is_web, dp.is_mobile, dp.is_enterprise,
                dp.is_advertising, dp.is_gamesvideo, dp.is_ecommerce, dp.is_biotech,
                dp.is_consulting
            });
        }

        // put these into function
        for (const auto& dp : trainingData) {
            X_train.push_back({
                dp.funding_total_usd, dp.funding_rounds, dp.age_first_funding_year,
                dp.relationships, dp.is_software, dp.is_web, dp.is_mobile, dp.is_enterprise,
                dp.is_advertising, dp.is_gamesvideo, dp.is_ecommerce, dp.is_biotech,
                dp.is_consulting
            });
            y_train.push_back(dp.is_successful);
        }

        for (const auto& dp : devData) {
            X_dev.push_back({
                dp.funding_total_usd, dp.funding_rounds, dp.age_first_funding_year,
                dp.relationships, dp.is_software, dp.is_web, dp.is_mobile, dp.is_enterprise,
                dp.is_advertising, dp.is_gamesvideo, dp.is_ecommerce, dp.is_biotech,
                dp.is_consulting
            });
            y_dev.push_back(dp.is_successful);
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
        //cout << "Total Data Points: " << dataset.data_points.size() << endl;
        //cout << "Training Data: " << trainingData.size() << ", Testing Data: " << testingData.size() << endl;
        //cout << "X_train: " << X_train.size() << ", y_train: " << y_train.size() << endl;
        
        model.w = vector<double>(X_train[0].size());
        for (double& weight : model.w) {
            weight = ((double)rand() / RAND_MAX) * 0.01; // Small random values
        }

        ////cout << "Before Normalization:" << endl;
        ////cout << "X_train size: " << X_train.size() << ", X_test size: " << X_test.size() << endl;

        int num_features = X_allData[0].size();
        vector<double> mean(num_features, 0.0);
        vector<double> stddev(num_features, 0.0);

        model.calc_mean_std(X_allData, mean, stddev);
        
        model.normalize_features(X_train, mean, stddev);
        model.normalize_features(X_dev, mean, stddev);
        model.normalize_features(X_test, mean, stddev);

        for (size_t i = 0; i < model.w.size(); ++i) {
            cout << "Weight[" << i << "]: " << model.w[i] << endl;
        }

        cout << "Bias (b): " << model.b << endl;


        ////cout << "After Normalization:" << endl;
        ////cout << "X_train size: " << X_train.size() << ", X_test size: " << X_test.size() << endl;

        // Train the model
        model.gradient_descent(X_train, y_train, model.w, model.b, model.num_iterations, model.learning_rate);

        vector<int> predictions_train = model.predict(X_train);

        int correct_train = 0;
        for (size_t i = 0; i < y_train.size(); ++i) {
            if (predictions_train[i] == y_train[i]) {
                ++correct_train;
            }
        }
        double accuracy_train = static_cast<double>(correct_train) / y_train.size();
        cout << "Training Accuracy: " << accuracy_train * 100 << "%" << endl;

        vector<int> predictions_dev = model.predict(X_dev);

        int correct_dev = 0;
        for (size_t i = 0; i < y_dev.size(); ++i) {
            if (predictions_dev[i] == y_dev[i]) {
                ++correct_dev;
            }
        }
        double accuracy_dev= static_cast<double>(correct_dev) / y_dev.size();
        cout << "Dev Accuracy: " << accuracy_dev * 100 << "%" << endl;

        // Predict and evaluate
        vector<int> predictions = model.predict(X_test);

        int correct = 0;
        for (size_t i = 0; i < y_test.size(); ++i) {
            if (predictions[i] == y_test[i]) {
                ++correct;
            }
        }
        double accuracy = static_cast<double>(correct) / y_test.size();
        cout << "Test Accuracy: " << accuracy * 100 << "%" << endl;

        char choice;
        do {
            cout << "\nWould you like to predict success for your startup? (y/n): ";
            cin >> choice;
            if (choice == 'y' || choice == 'Y') {
                predict_user_startup(model, mean, stddev);
            }
        } while (choice == 'y' || choice == 'Y');


    } else {
        cerr << "Failed to load data." << endl;
    }

    return 0;
}