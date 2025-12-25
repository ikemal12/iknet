#pragma once
#include <concepts>
#include <vector>
#include <span>
#include <type_traits>

namespace iknet {

template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
class Tensor {
public:
    Tensor(std::span<const T> data, std::vector<size_t> shape);

    int Tensor::calculate_size(const std::vector<int>& shape) const {
    
    }
private:
    std::vector<T> m_data;
    std::vector<size_t> m_shape; 
};

}