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
    : m_shape(std::move(shape)), m_offset(0) {
        size_t size = calculate_size(m_shape);
        if (data.size() != size) throw std::invalid_argument("Data size mismatch");

        m_data = std::make_shared<std::vector<T>>(data.begin(), data.end());
        m_strides = calculate_strides(m_shape);
    }

    Tensor(std::shared_ptr<std::vector<T>> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset)
    : m_data(data), m_shape(shape), m_strides(strides), m_offset(offset) {}

    [[nodiscard]] const std::vector<T>& data() const { return *m_data; }
    [[nodiscard]] const std::vector<size_t>& shape() const { return m_shape; }
    [[nodiscard]] const std::vector<size_t>& strides() const { return m_strides; }

    [[nodiscard]] static size_t calculate_size(const std::vector<size_t>& shape) {
        if (shape.empty()) return 0;
        size_t size = 1;
        for (size_t dim : shape) size *= dim;
        return size;
    }

    const T& at(const std::vector<size_t>& indices) const {
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

    [[nodiscard]] Tensor transpose(size_t dim0, size_t dim1) const {
        if (dim0 >= m_shape.size() || dim1 >= m_shape.size()) {
            throw std::out_of_range("Dimension index out of range");
        }
        std::vector<size_t> new_shape = m_shape;
        std::vector<size_t> new_strides = m_strides;
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);
        return Tensor(m_data, new_shape, new_strides, m_offset);
    }

    [[nodiscard]] Tensor broadcast_to(const std::vector<size_t>& target_shape) const {
        if (target_shape.size() < m_shape.size()) {
            throw std::invalid_argument("Cannot broadcast to fewer dimensions");
        }

        std::vector<size_t> new_strides(target_shape.size());
        size_t offset_diff = target_shape.size() - m_shape.size();

        for (size_t i = 0; i < target_shape.size(); ++i) {
            size_t dim = (i < offset_diff) ? 1 : m_shape[i - offset_diff];

            if (dim == target_shape[i]) {
                new_strides[i] = (i < offset_diff) ? 0 : m_strides[i - offset_diff];
            } else if (dim == 1) {
                new_strides[i] = 0;
            } else {
                throw std::invalid_argument("Shapes cannot be broadcast together");
            }
        }
        return Tensor(m_data, target_shape, new_strides, m_offset);
    }

    [[nodiscard]] bool is_contiguous() const {
        return m_strides == calculate_strides(m_shape);
    }

    [[nodiscard]] Tensor matmul(const Tensor& other) const {
        if (m_shape.size() != 2 || other.m_shape.size() != 2) {
            throw std::invalid_argument("Matmul currently only supports 2D matrices");
        }
        size_t M = m_shape[0];
        size_t K = m_shape[1];
        size_t K2 = other.m_shape[0];
        size_t N = other.m_shape[1];

        if (K != K2) {
            throw std::invalid_argument("Inner dimensions must match for matmul");
        }
        Tensor A = this->is_contiguous() ? *this: this->contiguous();
        Tensor B_T = other.transpose(0, 1).contiguous();
        const T* a_ptr = A.data().data();
        const T* b_ptr = B_T.data().data();
        std::vector<T> result_data(M*N);

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                T sum = 0;
                const T* row_a = a_ptr + (i * K);
                const T* row_b = b_ptr + (j * K);
                for(size_t k = 0; k < K; ++k) {
                    sum += row_a[k] * row_b[k];
                }
                result_data[i * N + j] = sum;
            }
        }
        return Tensor(result_data, {M, N});
    }

    [[nodiscard]] Tensor contiguous() const {
        if (is_contiguous()) {
            return Tensor(std::make_shared<std::vector<T>>(*m_data), m_shape, m_strides, m_offset);
        }

        size_t total = calculate_size(m_shape);
        std::vector<T> new_data;
        new_data.reserve(total);
        std::vector<size_t> index(m_shape.size(), 0);
        for (size_t i = 0; i < total; ++i) {
            size_t idx = m_offset;
            for(size_t dim = 0; dim < m_shape.size(); ++dim) {
                idx += index[dim] * m_strides[dim];
            }
            new_data.push_back((*m_data)[idx]);

            for (long dim = m_shape.size() - 1; dim >= 0; --dim) {
                index[dim]++;
                if (index[dim] < m_shape[dim]) break;
                index[dim] = 0;
            }
        }
        return Tensor(new_data, m_shape);
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
        std::vector<size_t> target_shape = infer_broadcast_shape(m_shape, other.m_shape);
        size_t total_elements = calculate_size(target_shape);
        std::vector<T> result_data;
        result_data.reserve(total_elements);

        if (m_shape == other.m_shape && this->is_contiguous() && other.is_contiguous()) {
            for (size_t i = 0; i < total_elements; ++i) {
                T val1 = (*m_data)[m_offset + i];
                T val2 = (*other.m_data)[other.m_offset + i];
                result_data.push_back(func(val1, val2));
            }
            return Tensor(result_data, target_shape);
        }

        Tensor lhs_broadcasted = this->broadcast_to(target_shape);
        Tensor rhs_broadcasted = other.broadcast_to(target_shape);
        std::vector<size_t> index(target_shape.size(), 0);

        for (size_t i = 0; i < total_elements; ++i) {
            size_t idx_lhs = lhs_broadcasted.m_offset;
            size_t idx_rhs = rhs_broadcasted.m_offset;

            for (size_t dim = 0; dim < target_shape.size(); ++dim) {
                idx_lhs += index[dim] * lhs_broadcasted.m_strides[dim];
                idx_rhs += index[dim] * rhs_broadcasted.m_strides[dim];
            }

            result_data.push_back(func((*lhs_broadcasted.m_data)[idx_lhs], (*rhs_broadcasted.m_data)[idx_rhs]));

            for (long dim = target_shape.size() - 1; dim >= 0; --dim) {
                index[dim]++;
                if (index[dim] < target_shape[dim]) break;
                index[dim] = 0;
            }
        }
        return Tensor(result_data, target_shape);
    }

    static std::vector<size_t> infer_broadcast_shape(const std::vector<size_t>& s1, const std::vector<size_t>& s2) {
        size_t dims1 = s1.size();
        size_t dims2 = s2.size();
        size_t n = std::max(dims1, dims2);
        std::vector<size_t> result(n);

        for (size_t i = 0; i < n; ++i) {
            size_t dim1 = (i < n - dims1) ? 1 : s1[i - (n - dims1)];
            size_t dim2 = (i < n - dims2) ? 1 : s2[i - (n - dims2)];
            
            if (dim1 == dim2) {
                result[i] = dim1;
            } else if (dim1 == 1) {
                result[i] = dim2;
            } else if (dim2 == 1) {
                result[i] = dim1;
            } else {
                throw std::invalid_argument("Shapes cannot be broadcast together");
            }
        }
        return result;
    }

};
} // namespace iknet