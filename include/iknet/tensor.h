#pragma once
#include <concepts>
#include <vector>
#include <span>
#include <type_traits>
#include <stdexcept>
#include <functional>
#include <memory>

namespace iknet {

template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
class Tensor {
public:
    Tensor(std::span<const T> data, std::vector<size_t> shape)
    : m_shape(std::move(shape)) m_offset(0) {
        size_t size = calculate_size(m_shape);
        if (data.size() != size) throw std::invalid_argument("Data size mismatch");

        m_data = std::make_shared<std::vector<T>>(data.begin(), data.end());
        m_strides = calculate_strides(m_shape);
    }

    Tensor(std::shared_ptr<std::vector<T>> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset)
    : m_data(data), m_shape(shape), m_strides(strides), m_offset(offset) {}

    const std::vector<size_t>& shape() const { return m_shape; }
    const std::vector<size_t>& strides() const { return m_strides; }

    static size_t calculate_size(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (size_t dim : shape) size *= dim;
        return size;
    }

    static std::vector<size_t> calculate_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        size_t current_stride = 1;
        for (long i = shape.size() - 1; i >= 0; --i) {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }
        return strides;
    }

    Tensor operator+(const Tensor& other) const { return elementwise_op(other, std::plus<T>()); }
    Tensor operator-(const Tensor& other) const { return elementwise_op(other, std::minus<T>()); }
    Tensor operator*(const Tensor& other) const { return elementwise_op(other, std::multiplies<T>()); }
    Tensor operator/(const Tensor& other) const { return elementwise_op(other, std::divides<T>()); }

private:
    // std::vector<T> m_data;
    std::shared_ptr<std::vector<T>> m_data;
    size_t m_offset = 0;
    std::vector<size_t> m_shape; 
    std::vector<size_t> m_strides;

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