#pragma once
#include <concepts>
#include <vector>
#include <span>
#include <type_traits>
#include <stdexcept>

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

    Tensor operator+(const Tensor& other) const {
        if (m_shape != other.m_shape) {
            throw std::invalid_argument("Shapes must match for addition");
        }
        std::vector<T> result_data;
        result_data.reserve(m_data.size());
        for (size_t i = 0; i < m_data.size(); ++i) {
            result_data.push_back(m_data[i] + other.m_data[i]);
        }
        return Tensor(result_data, m_shape);
    }

private:
    std::vector<T> m_data;
    std::vector<size_t> m_shape; 
};

}