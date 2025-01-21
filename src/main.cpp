#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
using namespace std;

struct DataPoint {

    int funding_total_usd; // y
    int funding_rounds; // y
    int age_first_funding_year;
    int relationships;
    int is_software; // y
    int is_web; // y
    int is_mobile; // y
    int is_enterprise; // y
    int is_advertising; // y
    int is_gamesvideo; // y
    int is_ecommerce; // y
    int is_biotech;
    int is_consulting;
    bool is_successful;  // Target: 1 if successful, 0 if failed
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
            stringstream ss(line); // treat the line string as an input stream so that we can extract each piece of data separated by commas
            DataPoint dp;

            getline(ss, line, ','); dp.funding_total_usd = stringToDouble(line);
            getline(ss, line, ','); dp.funding_rounds = stringToDouble(line);
            getline(ss, line, ','); dp.age_first_funding_year = stringToDouble(line);
            getline(ss, line, ','); dp.relationships = stringToDouble(line);

            // Encode binary flags (e.g., is_software, is_web, etc.)
            getline(ss, line, ','); dp.is_software = stringToDouble(line);
            getline(ss, line, ','); dp.is_web = stringToDouble(line);
            getline(ss, line, ','); dp.is_mobile = stringToDouble(line);
            getline(ss, line, ','); dp.is_enterprise = stringToDouble(line);
            getline(ss, line, ','); dp.is_advertising = stringToDouble(line);
            getline(ss, line, ','); dp.is_gamesvideo = stringToDouble(line);
            getline(ss, line, ','); dp.is_ecommerce = stringToDouble(line);
            getline(ss, line, ','); dp.is_biotech = stringToDouble(line);
            getline(ss, line, ','); dp.is_consulting = stringToDouble(line);

            // Let's assume the success column is "status", 1 = success, 0 = failure
            getline(ss, line, ',');
            dp.is_successful = (line == "active") ? 1 : 0; // "active" is considered successful

            data_points.push_back(dp);

        }

        file.close();
        return true;
    }

    vector<vector<double>> getFeatureMatrix(const vector<DataPoint>& data) {
        
    }

    // Printing all data
    void displayData() const {
        for (const auto &dp : data_points) {
            cout << ", Funding: $" << dp.funding_total_usd
                      << ", Rounds: " << dp.funding_rounds
                      << ", Age at First Funding: " << dp.age_first_funding_year
                      << ", Relationships: " << dp.relationships
                      << ", Software: " << dp.is_software
                      << ", Web: " << dp.is_web
                      << ", Success: " << (dp.is_successful ? "Yes" : "No")
                      << endl;
        }
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


int main() {
    Dataset dataset;

    // Vectors for training and testing data
    vector<DataPoint> trainingData;
    vector<DataPoint> testingData;

    // Feature matrix and labels vector
    vector<vector<double>> X;
    vector<int> y;

    if (dataset.loadFromCSV("startup_data.csv")) {
        dataset.splitData(trainingData, testingData);
        cout << "Data loaded and preprocessed successfully!" << endl;

        // Display dataset and feature matrix information
        dataset.displayData();
        cout << "Total Data Points: " << dataset.data_points.size() << endl;
        cout << "Training Data: " << trainingData.size() << ", Testing Data: " << testingData.size() << endl;

        if (!X.empty() && !X[0].empty()) {
            cout << "Number of examples: " << X.size() << endl;
            cout << "Number of features: " << X[0].size() << endl;
        } else {
            cerr << "Feature matrix is empty or improperly prepared." << endl;
        }
    } else {
        cerr << "Failed to load data." << endl;
    }

    return 0;
}