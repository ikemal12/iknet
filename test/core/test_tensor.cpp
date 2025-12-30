#include <gtest/gtest.h>
#include <iknet/core/tensor.h>
#include <vector>

TEST(TensorTest, InitializationAndAccess) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    iknet::Tensor<float> tensor(data, {2, 3});

    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
    EXPECT_EQ(tensor.strides()[0], 3);
    EXPECT_EQ(tensor.strides()[1], 1);
    EXPECT_FLOAT_EQ(tensor.at({1, 0}), 4.0f);
}
TEST(TensorTest, BasicArithmetic) {
    iknet::Tensor<float> a(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    iknet::Tensor<float> b(std::vector<float>{10, 20, 30, 40, 50, 60}, {2, 3});
    auto add = a + b;
    auto sub = a - b;
    auto mul = a * b;
    auto div = a / b;

    EXPECT_FLOAT_EQ(add.at({0, 0}), 11.0f); 
    EXPECT_FLOAT_EQ(sub.at({1, 2}), -54.0f); 
    EXPECT_FLOAT_EQ(mul.at({0, 1}), 40.0f); 
    EXPECT_FLOAT_EQ(div.at({1, 0}), 0.1f); 
}

TEST(TensorTest, TransposeAndStridedAdd) {
    iknet::Tensor<float> A(std::vector<float>{1, 2, 3, 4}, {2, 2});
    auto B = A.transpose(0, 1);
    EXPECT_EQ(B.shape()[0], 2);
    EXPECT_FLOAT_EQ(B.at({0, 1}), 3.0f);

    auto C = A + B;
    EXPECT_FLOAT_EQ(C.at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(C.at({0, 1}), 5.0f);
    EXPECT_FLOAT_EQ(C.at({1, 0}), 5.0f);
    EXPECT_FLOAT_EQ(C.at({1, 1}), 8.0f);
}

TEST(TensorTest, Broadcasting) {
    iknet::Tensor<float> m(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    iknet::Tensor<float> v(std::vector<float>{10, 20, 30}, {1, 3});
    auto result = m + v;

    EXPECT_FLOAT_EQ(result.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(result.at({1, 0}), 14.0f);
    EXPECT_FLOAT_EQ(result.at({1, 2}), 36.0f);
}

TEST(TensorTest, MatMul2D) {
    iknet::Tensor<float> A(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    iknet::Tensor<float> B(std::vector<float>{7, 8, 9, 1, 2, 3}, {3, 2});
    auto C = A.matmul(B);

    EXPECT_EQ(C.shape()[0], 2);
    EXPECT_EQ(C.shape()[1], 2);
    EXPECT_FLOAT_EQ(C.at({0, 0}), 31.0f);
    EXPECT_FLOAT_EQ(C.at({1, 1}), 55.0f);
}