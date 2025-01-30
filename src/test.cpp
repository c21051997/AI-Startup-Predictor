#include <gtest/gtest.h>

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


// Main function for running all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


