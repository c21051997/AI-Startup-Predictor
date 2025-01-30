#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include "DataPoint.h"

class Dataset {
public:
    std::vector<DataPoint> data_points;

    bool loadFromCSV(const std::string &filename);
    void displayData() const;
    void splitData(std::vector<DataPoint>& allData, std::vector<DataPoint>& trainingData,
                   std::vector<DataPoint>& devData, std::vector<DataPoint>& testingData);
    int stringToInt(const std::string& str);
};

#endif
