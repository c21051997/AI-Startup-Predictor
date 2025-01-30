#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>

bool Dataset::loadFromCSV(const std::string &filename) {
    std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }

        std::string line;
        getline(file, line);

        while (getline(file, line)){
            std::stringstream ss(line); // Stream for parsing the row
        DataPoint dp;

        std::string temp;

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

void Dataset::displayData() const {
    int success_count = 0;
    int failure_count = 0;

    for (const auto &dp : data_points) {

        if (dp.is_successful) success_count++;
        else failure_count++;

        std::cout << "Success: " << success_count << ", Failure: " << failure_count << std::endl;
            std::cout << ", Funding: $" << dp.funding_total_usd
                    << ", Rounds: " << dp.funding_rounds
                    << ", Age at First Funding: " << dp.age_first_funding_year
                    << ", Relationships: " << dp.relationships
                    << ", Software: " << dp.is_software
                    << ", Web: " << dp.is_web
                    << ", Success: " << (dp.is_successful ? "Yes" : "No")
                    << std::endl;
            
        }
        std::cout << "Success: " << success_count << ", Failure: " << failure_count << std::endl;
}

void Dataset::splitData(std::vector<DataPoint>& allData, std::vector<DataPoint>& trainingData,
                        std::vector<DataPoint>& devData, std::vector<DataPoint>& testingData) {
    double trainRatio = 0.6;
    double devRatio = 0.2;

    int trainSplitIndex = data_points.size() * trainRatio;
    int devSplitIndex = trainSplitIndex + data_points.size() * devRatio;

    std::random_device rd;         // Random device for seeding
    std::default_random_engine rng(rd()); 

    shuffle(data_points.begin(), data_points.end(), rng);

    trainingData.reserve(trainSplitIndex);                     // Allocate space for training data
    devData.reserve(devSplitIndex - trainSplitIndex);
    testingData.reserve(data_points.size() - devSplitIndex);

    copy(data_points.begin(), data_points.end(), back_inserter(allData));
    copy(data_points.begin(), data_points.begin() + trainSplitIndex, back_inserter(trainingData));
    copy(data_points.begin() + trainSplitIndex, data_points.begin() + devSplitIndex, back_inserter(devData));
    copy(data_points.begin() + devSplitIndex, data_points.end(), back_inserter(testingData));
}


int Dataset::stringToInt(const std::string& str) {
        return std::stoi(str); // Can throw exceptions for invalid input
}