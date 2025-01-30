# Logistic Regression Project

This project implements a logistic regression model from scratch, trained and tested on a real-world dataset sourced from Kaggle. The goal is to classify data points based on input features and evaluate the performance of the model using accuracy metrics. Additionally, Google Test (GTest) is integrated for unit testing key components of the implementation.

---

## Features
- **Logistic Regression Implementation**: Custom implementation of logistic regression with gradient descent.
- **Dataset Handling**: Includes functionality to load and preprocess a dataset from a CSV file.
- **Evaluation Metrics**: Training, development (validation), and test set accuracy tracking.
- **Unit Testing**: Utilizes Google Test to ensure correctness of various functions and components.

---

## Prerequisites
To run this project, ensure you have the following installed:

- **C++ Compiler** (e.g., g++)
- **CMake**
- **Google Test Library**

---

## Project Structure
```
|-- LogisticRegression.h/cpp      # Logistic regression implementation
|-- Dataset.h/cpp                 # Dataset loading and preprocessing
|-- main.cpp                      # Main program to train and test the model
|-- tests.cpp                     # Google Test cases
|-- CMakeLists.txt                # Build configuration
|-- data/                         # Directory for dataset (CSV file)
|-- README.md                     # Project documentation (this file)
```

---

## How to Run the Project

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Prepare the Dataset
- Navigate to the src file:
  ```bash
  cd src
  ```
- Complie the project using the following command:
  ```bash
  g++ -o startup_predictor main.cpp Dataset.cpp LogisticRegression.cpp -std=c++17
  ```

### Step 3: Build the Project
- Run the project using the following command:
   ```bash
   ./main
   ```

### Expected Output
The program will print the following:
- Training, development, and test accuracies.
- Optionally, you can enable gradient computation or other debug outputs in the source code.

---

## How to Run Google Tests

### Step 1: Build Tests
Ensure GTest is linked correctly in your `CMakeLists.txt`. From the `build/` directory:
```bash
make
```

### Step 2: Execute Tests
Run the test binary:
```bash
./MyTests
```

### Expected Output
Google Test will display results for each test case, e.g.:
```
[==========] Running 5 tests from 2 test cases.
[----------] Global test environment set-up.
[----------] 3 tests from AdditionTest
[ RUN      ] AdditionTest.HandlesPositiveNumbers
[       OK ] AdditionTest.HandlesPositiveNumbers (0 ms)
...
[==========] 5 tests from 2 test cases ran. (2 ms total)
[  PASSED  ] 5 tests.
```

---

## Potential Improvements
- Add k-fold cross-validation for better model evaluation.
- Experiment with feature scaling and regularization techniques to improve accuracy.
- Enhance data preprocessing by adding more feature engineering steps.

---

## Acknowledgments
- Dataset sourced from [Kaggle](https://www.kaggle.com/).
- Google Test library for unit testing.

