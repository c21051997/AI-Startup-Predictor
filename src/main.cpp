#include <iostream>
#include "Dataset.h"
#include "LogisticRegression.h"
#include "DataPoint.h"


void predict_user_startup(LogisticRegression& model, std::vector<double>& mean, std::vector<double>&stddev) {
    std::vector<double> user_input;

    std::cout << "Enter details about your startup:" << std::endl;
    double funding_total_usd, funding_rounds, age_first_funding_year, relationships;
    double is_software, is_web, is_mobile, is_enterprise, is_advertising, is_gamesvideo, is_ecommerce, is_biotech, is_consulting;

    std::cout << "Funding Total (USD): ";
    std::cin >> funding_total_usd;
    std::cout << "Funding Rounds: ";
    std::cin >> funding_rounds;
    std::cout << "Age at First Funding Year: ";
    std::cin >> age_first_funding_year;
    std::cout << "Number of Relationships: ";
    std::cin >> relationships;

    std::cout << "Is your company Software-related? (1 for Yes, 0 for No): ";
    std::cin >> is_software;
    std::cout << "Is it Web-based? (1 for Yes, 0 for No): ";
    std::cin >> is_web;
    std::cout << "Is it Mobile-focused? (1 for Yes, 0 for No): ";
    std::cin >> is_mobile;
    std::cout << "Is it Enterprise-oriented? (1 for Yes, 0 for No): ";
    std::cin >> is_enterprise;
    std::cout << "Is it focused on Advertising? (1 for Yes, 0 for No): ";
    std::cin >> is_advertising;
    std::cout << "Is it in Games/Video? (1 for Yes, 0 for No): ";
    std::cin >> is_gamesvideo;
    std::cout << "Is it in E-commerce? (1 for Yes, 0 for No): ";
    std::cin >> is_ecommerce;
    std::cout << "Is it in Biotech? (1 for Yes, 0 for No): ";
    std::cin >> is_biotech;
    std::cout << "Is it a Consulting company? (1 for Yes, 0 for No): ";
    std::cin >> is_consulting;

    // Prepare user input std::vector for continuous and categorical features separately
    std::vector<double> continuous_input = {
        funding_total_usd, funding_rounds, age_first_funding_year, relationships
    };
    std::vector<double> categorical_input = {
        is_software, is_web, is_mobile, is_enterprise,
        is_advertising, is_gamesvideo, is_ecommerce, is_biotech, is_consulting
    };

    // Normalize continuous features only
    std::vector<std::vector<double>> continuous_input_vector = {continuous_input};
    model.normalize_features(continuous_input_vector, mean, stddev);

    // Combine normalized continuous features with original categorical features
    user_input.clear();
    user_input.insert(user_input.end(), continuous_input_vector[0].begin(), continuous_input_vector[0].end());
    user_input.insert(user_input.end(), categorical_input.begin(), categorical_input.end());

    std::cout << "Normalized user input: ";
    for (const auto& feature : user_input) {
        std::cout << feature << " ";  // This will print each normalized feature
    }
    std::cout << std::endl;

    // Predict success probability
    double probability = model.sigmoid(model.dot_product(user_input, model.w) + model.b);

    // Output the result
    std::cout << "Your startup's predicted success probability is: " << probability * 100 << "%" << std::endl;
    if (probability >= 0.5) {
        std::cout << "Prediction: Likely to Succeed!" << std::endl;
    } else {
        std::cout << "Prediction: Less likely to Succeed." << std::endl;
    }
}


int main() {
    Dataset dataset;
    LogisticRegression model(0.1, 100);

    // std::vectors for training and testing data
    std::vector<DataPoint> allData;
    std::vector<DataPoint> trainingData;
    std::vector<DataPoint> devData;
    std::vector<DataPoint> testingData;

    if (dataset.loadFromCSV("/Users/harridavies/Desktop/Startup Predictor/src/startup_data.csv")) {
        //dataset.displayData();
        dataset.splitData(allData, trainingData, testingData, devData);
        //std::cout << "Data loaded and preprocessed successfully!" << std::endl;

        std::vector<std::vector<double> > X_allData;
        // Step 2: Prepare feature matrix (X) and labels (y)
        std::vector<std::vector<double> > X_train;
        std::vector<int> y_train;

        std::vector<std::vector<double> > X_dev;
        std::vector<int> y_dev;

        // Prepare testing data
        std::vector<std::vector<double> > X_test;
        std::vector<int> y_test;

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
        //std::cout << "Total Data Points: " << dataset.data_points.size() << std::endl;
        //std::cout << "Training Data: " << trainingData.size() << ", Testing Data: " << testingData.size() << std::endl;
        //std::cout << "X_train: " << X_train.size() << ", y_train: " << y_train.size() << std::endl;
        
        model.w = std::vector<double>(X_train[0].size());
        for (double& weight : model.w) {
            weight = ((double)rand() / RAND_MAX) * 0.01; // Small random values
        }

        ////std::cout << "Before Normalization:" << std::endl;
        ////std::cout << "X_train size: " << X_train.size() << ", X_test size: " << X_test.size() << std::endl;

        int num_features = X_allData[0].size();
        std::vector<double> mean(num_features, 0.0);
        std::vector<double> stddev(num_features, 0.0);

        model.calc_mean_std(X_allData, mean, stddev);
        
        model.normalize_features(X_train, mean, stddev);
        model.normalize_features(X_dev, mean, stddev);
        model.normalize_features(X_test, mean, stddev);

        for (size_t i = 0; i < model.w.size(); ++i) {
            // std::cout << "Weight[" << i << "]: " << model.w[i] << std::endl;
        }

        // std::cout << "Bias (b): " << model.b << std::endl;


        ////std::cout << "After Normalization:" << std::endl;
        ////std::cout << "X_train size: " << X_train.size() << ", X_test size: " << X_test.size() << std::endl;

        // Train the model
        model.gradient_descent(X_train, y_train, model.w, model.b, model.num_iterations, model.learning_rate);

        std::vector<int> predictions_train = model.predict(X_train);

        int correct_train = 0;
        for (size_t i = 0; i < y_train.size(); ++i) {
            if (predictions_train[i] == y_train[i]) {
                ++correct_train;
            }
        }
        double accuracy_train = static_cast<double>(correct_train) / y_train.size();
        std::cout << "Training Accuracy: " << accuracy_train * 100 << "%" << std::endl;

        std::vector<int> predictions_dev = model.predict(X_dev);

        int correct_dev = 0;
        for (size_t i = 0; i < y_dev.size(); ++i) {
            if (predictions_dev[i] == y_dev[i]) {
                ++correct_dev;
            }
        }
        double accuracy_dev= static_cast<double>(correct_dev) / y_dev.size();
        std::cout << "Dev Accuracy: " << accuracy_dev * 100 << "%" << std::endl;

        // Predict and evaluate
        std::vector<int> predictions = model.predict(X_test);

        int correct = 0;
        for (size_t i = 0; i < y_test.size(); ++i) {
            if (predictions[i] == y_test[i]) {
                ++correct;
            }
        }
        double accuracy = static_cast<double>(correct) / y_test.size();
        std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;

        char choice;
        do {
            std::cout << "\nWould you like to predict success for your startup? (y/n): ";
            std::cin >> choice;
            if (choice == 'y' || choice == 'Y') {
                predict_user_startup(model, mean, stddev);
            }
        } while (choice == 'y' || choice == 'Y');


    } else {
        std::cerr << "Failed to load data." << std::endl;
    }

    return 0;
}