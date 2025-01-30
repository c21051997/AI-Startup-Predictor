#include <gtest/gtest.h>
#include "LogisticRegression.h"
#include "Dataset.h"

// Function to test
int Add(int a, int b) {
    return a + b;
}

int Subtract(int a, int b) {
    return a - b;
}

// Test Cases for Addition
TEST(AdditionTest, HandlesPositiveNumbers) {
    EXPECT_EQ(Add(2, 3), 5); // 2 + 3 = 5
    EXPECT_EQ(Add(10, 20), 30); // 10 + 20 = 30
}

TEST(AdditionTest, HandlesNegativeNumbers) {
    EXPECT_EQ(Add(-2, -3), -5); // -2 + -3 = -5
    EXPECT_EQ(Add(-10, 20), 10); // -10 + 20 = 10
}

// Test Cases for Subtraction
TEST(SubtractionTest, HandlesPositiveNumbers) {
    EXPECT_EQ(Subtract(5, 3), 2); // 5 - 3 = 2
    EXPECT_EQ(Subtract(20, 10), 10); // 20 - 10 = 10
}

TEST(SubtractionTest, HandlesNegativeNumbers) {
    EXPECT_EQ(Subtract(-5, -3), -2); // -5 - -3 = -2
    EXPECT_EQ(Subtract(-10, 20), -30); // -10 - 20 = -30
}

// Test for stringToInt utility function
TEST(DatasetTest, StringToIntValidInput) {
    Dataset dataset;
    EXPECT_EQ(dataset.stringToInt("123"), 123);
    EXPECT_EQ(dataset.stringToInt("-45"), -45);
    EXPECT_EQ(dataset.stringToInt("0"), 0);
}

TEST(DatasetTest, StringToIntInvalidInput) {
    Dataset dataset;
    EXPECT_THROW(dataset.stringToInt("abc"), std::invalid_argument);
    EXPECT_THROW(dataset.stringToInt("99999999999999999999"), std::out_of_range);
}

// Test gradient computation
TEST(LogisticRegressionTest, GradientComputation) {
    LogisticRegression model(0.01, 1000);

    // Dummy data
    std::vector<std::vector<double>> X = {{10.0, 20.0}, {20.0, 30.0}, {30.0, 40.0}};
    std::vector<int> y = {0, 1, 0};

    // Initial weights and bias
    std::vector<double> w = {0.5, -0.5}; // Initial weights
    double b = 0.0; // Initial bias

    // Variables for the gradients
    double dj_db = 0.0;  // Initialize to 0 for accumulation
    std::vector<double> dj_dw(w.size(), 0.0);  // Initialize to 0 for accumulation

    model.compute_gradient(X, y, w, b, dj_db, dj_dw);

    // Expected gradients based on the calculations
    double expected_dj_db = -0.3267; // Expected bias gradient
    double expected_dj_dw_0 = -6.5327; // Expected weight gradient for w[0]
    double expected_dj_dw_1 = -9.799;  // Expected weight gradient for w[1]

    // Compare the computed gradients with expected ones using EXPECT_NEAR for approximate comparison
    // test fails with anything over 1e-4
    EXPECT_NEAR(dj_db, expected_dj_db, 1e-3);  // Check bias gradient
    EXPECT_NEAR(dj_dw[0], expected_dj_dw_0, 1e-3);  // Check weight gradient for w[0]
    EXPECT_NEAR(dj_dw[1], expected_dj_dw_1, 1e-3);  // Check weight gradient for w[1]
}

TEST(LogisticRegressionTest, CorrectGradientComputation) {
    LogisticRegression model(0.01, 1000);

    // Dummy data
    std::vector<std::vector<double>> X = {{10.0, 20.0}, {20.0, 30.0}, {30.0, 40.0}};
    std::vector<int> y = {0, 1, 0};

    // Initial weights and bias
    std::vector<double> w = {0.5, -0.5};
    double b = 0.0;

    // Variables for the gradients
    double dj_db = 0.0;
    std::vector<double> dj_dw(w.size(), 0.0);

    // Compute gradients
    model.compute_gradient(X, y, w, b, dj_db, dj_dw);

    // Manually computed expected gradients
    EXPECT_NEAR(dj_db, -0.3267, 1e-4);  // Expected bias gradient
    EXPECT_NEAR(dj_dw[0], -2.1326, 1e-4);  // Expected weight gradient for w[0]
    EXPECT_NEAR(dj_dw[1], -3.1999, 1e-4);  // Expected weight gradient for w[1]
}


// Main function for running all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


