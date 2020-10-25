#ifndef FEATURE_VECTOR_HPP
#define FEATURE_VECTOR_HPP

#include <functional>
#include <vector>

#include "structures.hpp"

template<class Iter>
auto get_polynomial_feature_vector(Iter start, Iter end, int max_degree, std::vector<double>& acc) {
    if (start == end) {
        return acc;
    }
    else {
        auto val = *start;
        for (auto x : acc) {
            for (auto degree = 1; degree <= max_degree; degree++) {
                acc.emplace_back(x * std::pow(val, degree));
            }
        }
        return get_polynomial_feature_vector(start + 1, end, max_degree, acc);
    }
}

template<class Iter>
auto get_polynomial_feature_vector(Iter start, Iter end, int max_degree) {
    std::vector<double> acc {1};
    return get_polynomial_feature_vector(start, end, max_degree, acc);
}

template<class Iterable>
auto get_polynomial_feature_vector(Iterable iterable, int max_degree) {
    return get_polynomial_feature_vector(iterable.begin(), iterable.end(), max_degree);
}

template<class Iterable1, class Iterable2>
std::function<FeatureVector(InstanceSize)> standardize(std::function<FeatureVector(InstanceSize)> feature_func, Iterable1 means, Iterable2 stddevs) {
    return [feature_func, means, stddevs](InstanceSize instance_size) {
        auto raw = feature_func(instance_size);
        auto iter = raw.begin();
        auto means_iter = means.begin();
        auto stddevs_iter = stddevs.begin();
        // Skip intercept.
        iter++;
        means_iter++;
        stddevs_iter++;
        for (; iter != raw.end(); iter++) {
            *iter = (*iter - *means_iter) / (*stddevs_iter);
            means_iter++;
            stddevs_iter++;
        }
        return raw;
    };
}

#endif