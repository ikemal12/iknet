#include <iknet/tensor.h>
#include <vector>
#include <iostream>
#include <span>

namespace iknet {
}


int main() {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    std::vector<size_t> shape = {2, 3};
    iknet::Tensor<float> tensor(data, shape);

    for (auto dim : tensor.shape()) std::cout << dim << " ";
    for (auto val : tensor.data()) std::cout << val << " ";

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
    return 0;
}