#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace sam3d {
// Owned, immutable snapshot. Unlike remembering a pointer to a mutable vector,
// this value cannot silently refer to changed/unvalidated bytes on reuse.
class validated_weights {
public:
    validated_weights():validated_weights(std::vector<float>{},checked_tag{}){}
    validated_weights(std::span<const float> values):validated_weights(std::vector<float>(values.begin(),values.end())){}
    static validated_weights checked(std::vector<float> values) {
        return validated_weights(std::move(values));
    }
    // Implicit conversion lets diagnostic readers provide owned vectors while
    // ensuring they enter the same checked/immutable parameter contract.
    validated_weights(std::vector<float> values) : validated_weights(std::move(values),checked_tag{}) {
        if (!std::all_of(data_->begin(),data_->end(),[](float x){return std::isfinite(x);}))
            throw std::invalid_argument("non-finite immutable weights");
    }
    std::span<const float> values() const { return *data_; }
    // Const range interface: graph builders can consume immutable parameters
    // without materializing a fresh vector. No mutable storage is exposed.
    size_t size() const {return data_->size();}
    bool empty() const {return data_->empty();}
    const float *data() const {return data_->data();}
    auto begin() const {return data_->begin();}
    auto end() const {return data_->end();}
    const float &operator[](size_t i) const {return (*data_)[i];}
private:
    friend class tensor_archive;
    friend class neural_session;
    struct checked_tag {};
    validated_weights(std::vector<float> values,checked_tag)
        : data_(std::make_shared<const std::vector<float>>(std::move(values))) {}
    std::shared_ptr<const std::vector<float>> data_;
};
}
