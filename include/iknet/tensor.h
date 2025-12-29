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
concept Numeric = ::std::is_arithmetic_v<T>;

template<Numeric T>
class Tensor {
public:
    Tensor(std::span<const T> data, std::vector<size_t> shape)
    : m_shape(std::move(shape)), m_offset(0) {
        size_t size = calculate_size(m_shape);
        if (data.size() != size) throw std::invalid_argument("Data size mismatch");

        m_data = std::make_shared<std::vector<T>>(data.begin(), data.end());
        m_strides = calculate_strides(m_shape);
    }

    Tensor(std::shared_ptr<std::vector<T>> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset)
    : m_data(data), m_shape(shape), m_strides(strides), m_offset(offset) {}

    const std::vector<T>& data() const { return *m_data; }
    const std::vector<size_t>& shape() const { return m_shape; }
    const std::vector<size_t>& strides() const { return m_strides; }

    static size_t calculate_size(const std::vector<size_t>& shape) {
        if (shape.empty()) return 0;
        size_t size = 1;
        for (size_t dim : shape) size *= dim;
        return size;
    }

    T& at(const std::vector<size_t>& indices) {
        size_t flat_index = m_offset;
        for (size_t i = 0; i < indices.size(); ++i) {
            flat_index += indices[i] * m_strides[i];
        }
        return (*m_data)[flat_index];
    }

    Tensor operator+(const Tensor& other) const { return elementwise_op(other, std::plus<T>()); }
    Tensor operator-(const Tensor& other) const { return elementwise_op(other, std::minus<T>()); }
    Tensor operator*(const Tensor& other) const { return elementwise_op(other, std::multiplies<T>()); }
    Tensor operator/(const Tensor& other) const { return elementwise_op(other, std::divides<T>()); }

    Tensor transpose(size_t dim0, size_t dim1) const {
        if (dim0 >= m_shape.size() || dim1 >= m_shape.size()) {
            throw std::out_of_range("Dimension index out of range");
        }
        std::vector<size_t> new_shape = m_shape;
        std::vector<size_t> new_strides = m_strides;
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);
        return Tensor(m_data, new_shape, new_strides, m_offset);
    }

private:
    std::shared_ptr<std::vector<T>> m_data;
    std::vector<size_t> m_shape; 
    std::vector<size_t> m_strides;
    size_t m_offset = 0;

    static std::vector<size_t> calculate_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        size_t current_stride = 1;
        for (long i = shape.size() - 1; i >= 0; --i) {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }
        return strides;
    }

    template<typename Func>
    Tensor elementwise_op(const Tensor& other, Func func) const {
        if (m_shape != other.m_shape) {
                throw std::invalid_argument("Shapes must match for element-wise operation");
        }
        
        size_t total_elements = calculate_size(m_shape);
        std::vector<T> result_data;
        result_data.reserve(total_elements);

        std::vector<size_t> index(m_shape.size(), 0);

        for (size_t i = 0; i < total_elements; ++i) {
            size_t idx_self = m_offset;
            size_t idx_other = other.m_offset;

            for (size_t dim = 0; dim < m_shape.size(); ++dim) {
                idx_self += index[dim] * m_strides[dim];
                idx_other += index[dim] * other.m_strides[dim];
            }

            result_data.push_back(func((*m_data)[idx_self], (*other.m_data)[idx_other]));

            for (long dim = m_shape.size() - 1; dim >= 0; --dim) {
                index[dim]++;
                if (index[dim] < m_shape[dim]) break;
                index[dim] = 0;
            }
        }
        return Tensor(result_data, m_shape);
    }
};