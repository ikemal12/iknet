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
    return 0;
}