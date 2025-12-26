#pragma once
#include <concepts>
#include <vector>
#include <span>
#include <type_traits>
#include <stdexcept>
#include <functional>

namespace iknet {

template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
class Tensor {
public:
    Tensor(std::span<const T> data, std::vector<size_t> shape)
    : m_data(data.begin(), data.end()), m_shape(std::move(shape)) {}

    Tensor(const std::vector<T>& data, std::vector<size_t> shape)
    : Tensor(std::span<const T>(data), std::move(shape)) {}

    const std::vector<size_t>& shape() const { return m_shape; }
    const std::vector<T>& data() const { return m_data; }

    static size_t calculate_size(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (size_t dim : shape) size *= dim;
        return size;
    }

    Tensor operator+(const Tensor& other) const { return elementwise_op(other, std::plus<T>()); }
    Tensor operator-(const Tensor& other) const { return elementwise_op(other, std::minus<T>()); }
    Tensor operator*(const Tensor& other) const { return elementwise_op(other, std::multiplies<T>()); }
    Tensor operator/(const Tensor& other) const { return elementwise_op(other, std::divides<T>()); }

private:
    std::vector<T> m_data;
    std::vector<size_t> m_shape; 

    template<typename Func>
    Tensor elementwise_op(const Tensor& other, Func func) const {
        if (m_shape != other.m_shape) {
                throw std::invalid_argument("Shapes must match for element-wise operation");
            }
            std::vector<T> result_data;
            result_data.reserve(m_data.size());
            for (size_t i = 0; i < m_data.size(); ++i) {
                result_data.push_back(func(m_data[i], other.m_data[i]));
            }
            return Tensor(result_data, m_shape);
    }
};

}