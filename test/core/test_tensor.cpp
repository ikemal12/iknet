#include <iknet/tensor.h>
#include <vector>
#include <iostream>
#include <span>
#include <cassert>

namespace iknet {
}

int main() {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    std::vector<size_t> shape = {2, 3};
    iknet::Tensor<float> tensor(data, shape);

    for (auto dim : tensor.shape()) std::cout << dim << " ";
    for (auto val : tensor.data()) std::cout << val << " ";

    float val = tensor.at({1, 0});
    std::cout << "Value at (1, 0): " << val << std::endl;

    iknet::Tensor<float> a(std::span<const float>({1, 2, 3, 4, 5, 6}), {2, 3});
    iknet::Tensor<float> b(std::span<const float>({10, 20, 30, 40, 50, 60}), {2, 3});
    auto c_add = a + b;
    auto c_sub = a - b;
    auto c_mul = a * b;
    auto c_div = a / b;
    for (auto val : c_add.data()) std::cout << val << " "; 
    for (auto val : c_sub.data()) std::cout << val << " "; 
    for (auto val : c_mul.data()) std::cout << val << " "; 
    for (auto val : c_div.data()) std::cout << val << " "; 

    // 1. SETUP: Create Matrix A (2x2)
    // [ 1, 2 ]
    // [ 3, 4 ]
    std::vector<float> data2 = {1, 2, 3, 4};
    iknet::Tensor<float> x(data2, {2, 2});

    // 2. CREATE TRANSPOSE VIEW 'B' OF A
    // [ 1, 3 ]
    // [ 2, 4 ]
    auto y = x.transpose(0, 1);

    // 3. ADD THEM 
    // We are adding a contiguous tensor (A) to a non-contiguous view (B).
    // [ 1, 2 ]   [ 1, 3 ]   [ 2, 5 ]
    // [ 3, 4 ] + [ 2, 4 ] = [ 5, 8 ]
    auto c = x + y;
    std::cout << "Result of A + A.T:\n";
    std::cout << c.at({0, 0}) << " (Expected 2)" << std::endl;
    assert(c.at({0, 0}) == 2.0f);
    std::cout << c.at({0, 1}) << " (Expected 5)" << std::endl;
    assert(c.at({0, 1}) == 5.0f);
    std::cout << c.at({1, 1}) << " (Expected 8)" << std::endl;
    assert(c.at({1, 1}) == 8.0f);

    iknet::Tensor<float> d(std::vector<float>{10, 20, 30}, {1, 3});
    auto e = tensor + d;
    std::cout << "Broadcast Result (0,0): " << e.at({0, 0}) << " (Expected 11)" << std::endl;
    std::cout << "Broadcast Result (1,0): " << e.at({1, 0}) << " (Expected 14)" << std::endl;
    assert(e.at({0, 0}) == 11);
    assert(e.at({1, 0}) == 14);

    iknet::Tensor<float> f(std::vector<float>{7, 8, 9, 1, 2, 3}, {3, 2});
    auto g = tensor.matmul(f);
    // VERIFY:
    // Calculation for C[0,0]:
    // Row 0 of A: [1, 2, 3]
    // Col 0 of B: [7, 9, 2]
    // Dot Prod: (1*7) + (2*9) + (3*2) = 7 + 18 + 6 = 31
    std::cout << "MatMul [0,0]: " << g.at({0, 0}) << " (Expected 31)" << std::endl;
    assert(g.at({0, 0}) == 31);

    // Calculation for C[1,1]:
    // Row 1 of A: [4, 5, 6]
    // Col 1 of B: [8, 1, 3]
    // Dot Prod: (4*8) + (5*1) + (6*3) = 32 + 5 + 18 = 55
    std::cout << "MatMul [1,1]: " << g.at({1, 1}) << " (Expected 55)" << std::endl;
    assert(g.at({1, 1}) == 55);

    std::cout << "SUCCESS: Matrix Multiplication Verified." << std::endl;

    return 0;
}