#include <iknet/tensor.h>
#include <vector>
#include <iostream>

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
    auto c = a + b;
    for (auto val : c.data()) std::cout << val << " "; 

    return 0;
}