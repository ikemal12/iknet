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
    Tensor(std::span<const T> data, std::vector<size_t> shape)
    : m_data(data.begin(), data.end()), m_shape(std::move(shape)) {}

    static size_t calculate_size(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (size_t dim : shape) size *= dim;
        return size;
    }

    const std::vector<size_t>& shape() const { return m_shape; }
    const std::vector<T>& data() const { return m_data; }
    
private:
    std::vector<T> m_data;
    std::vector<size_t> m_shape; 
};

}