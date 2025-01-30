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

// Main function for running all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


