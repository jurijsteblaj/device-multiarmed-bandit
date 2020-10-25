#ifndef DATASET_HPP
#define DATASET_HPP

#include <array>
#include <iostream>
#include <iterator>
#include <vector>

#include <stan/math.hpp>

#include "structures.hpp"

class Experiment {
private:
    InstanceSize _instance_size;
    std::array<Duration,DEVICE_COUNT> _processing_time;
public:
    Experiment(const InstanceSize instance_size, const std::array<Duration,2> processing_time)
        : _instance_size(instance_size), _processing_time(processing_time) {}
    
    Experiment(const size_t instance_size_size) {
        _instance_size.resize(instance_size_size);
    }
    
    auto get_instance_size() const { return _instance_size; };
    auto get_processing_time() const { return _processing_time; };
    auto get_processing_time(Device device) const { return _processing_time[device]; };

    friend std::ostream& operator<<(std::ostream& out, const Experiment experiment) {
    return out << experiment.get_instance_size() << ","
        << experiment.get_processing_time(CPU) << ","
        << experiment.get_processing_time(GPU);
    }

    friend std::istream& operator>>(std::istream& in, Experiment& experiment) {
        char ch;
        in >> experiment._instance_size >> ch 
            >> experiment._processing_time[CPU] >> ch
            >> experiment._processing_time[GPU];
        return in; 
    }
};

stan::math::matrix_d multiply(const stan::math::matrix_d& a, const stan::math::matrix_d& b) {
    auto c = (a * b).eval();
    return c;
}

stan::math::matrix_d multiply_cl(const stan::math::matrix_d& a, const stan::math::matrix_d& b) {
    stan::math::matrix_cl<double> a_cl(a);
    stan::math::matrix_cl<double> b_cl(b);

    auto c_cl = a_cl * b_cl;

    auto c_from_cl = stan::math::from_matrix_cl(c_cl).eval();

    return c_from_cl;
}

stan::math::matrix_d cholesky_decompose(const stan::math::matrix_d& a) {
    auto c = stan::math::cholesky_decompose(a).eval();
    return c;
}

// template <typename T>
// Eigen::Matrix<T, -1, -1> cholesky_decompose(const Eigen::Matrix<T, -1, -1>& a) {
//     auto c = stan::math::cholesky_decompose(a).eval();
//     return c;
// }

stan::math::matrix_d cholesky_decompose_cl(const stan::math::matrix_d& a) {
    stan::math::matrix_cl<double> a_cl(a);

    stan::math::opencl::cholesky_decompose(a_cl);

    auto c_from_cl = stan::math::from_matrix_cl(a_cl).eval();

    return c_from_cl;
}

// template <typename T>
// Eigen::Matrix<T, -1, -1> cholesky_decompose_cl(const Eigen::Matrix<T, -1, -1>& a) {
//     stan::math::matrix_cl<double> a_cl(a);

//     stan::math::opencl::cholesky_decompose(a_cl);

//     auto c_from_cl = stan::math::from_matrix_cl(a_cl).eval();

//     return c_from_cl;
// }

stan::math::matrix_d multiply_choice(const stan::math::matrix_d& a, const stan::math::matrix_d& b, Device choice) {
    if (choice == CPU) {
        return multiply(a, b);
    }
    else {
        return multiply_cl(a, b);
    }
}

std::vector<std::array<stan::math::matrix_d,2>> generate_problem_instances(const size_t repetition_count) {
    std::vector<std::array<stan::math::matrix_d,2>> problem_instances;
    for (size_t j = 50; j < 5'000; j *= 2) {
        auto k = j;
        auto l = j;

        auto a_rng = stan::math::matrix_d::Random(j, k);
        auto b_rng = stan::math::matrix_d::Random(k, l);

        for (auto repetition = 0; repetition < repetition_count; repetition++) {
            std::array<stan::math::matrix_d,2> problem_instance {
                stan::math::matrix_d(a_rng),
                stan::math::matrix_d(b_rng)
            };

            problem_instances.emplace_back(std::move(problem_instance));
        }
    }

    return problem_instances;
}

std::vector<std::array<stan::math::matrix_d,2>> generate_problem_instances(const std::vector<size_t> sizes) {
    std::vector<std::array<stan::math::matrix_d,2>> problem_instances;
    for (auto j : sizes) {
        auto k = j;
        auto l = j;

        auto a_rng = stan::math::matrix_d::Random(j, k);
        auto b_rng = stan::math::matrix_d::Random(k, l);

        std::array<stan::math::matrix_d,2> problem_instance {
            stan::math::matrix_d(a_rng),
            stan::math::matrix_d(b_rng)
        };

        problem_instances.emplace_back(std::move(problem_instance));
    }

    return problem_instances;
}

std::vector<Experiment> generate_experiments(const size_t repetition_count) {
    std::vector<Experiment> experiments;
    for (size_t j = 50; j < 5'000; j *= 2) {
        auto k = j;
        auto l = j;

        auto a_rng = stan::math::matrix_d::Random(j, k);
        auto b_rng = stan::math::matrix_d::Random(k, l);

        for (auto repetition = 0; repetition < repetition_count; repetition++) {
            std::cerr << "j = " << j << ", rep = " << repetition + 1 << std::endl;
            stan::math::matrix_d a(a_rng);
            stan::math::matrix_d b(b_rng);
            const auto cpu_start = Clock::now();
            volatile auto c = multiply(a, b);
            const auto cpu_end = Clock::now();

            const auto gpu_start = Clock::now();
            volatile auto c_from_cl = multiply_cl(a, b);
            const auto gpu_end = Clock::now();
            
            experiments.emplace_back(InstanceSize {j, k, l},
                std::array<Duration,2> { cpu_end - cpu_start, gpu_end - gpu_start});
        }
    }

    return experiments;
}


std::vector<Experiment> generate_experiments(const std::vector<size_t> sizes) {
    std::vector<Experiment> experiments;

    for (auto j : sizes) {
        auto k = j;
        auto l = j;

        auto a_rng = stan::math::matrix_d::Random(j, k);
        auto b_rng = stan::math::matrix_d::Random(k, l);
        //std::cerr << "j = " << j << ", rep = " << 1 << std::endl;
        stan::math::matrix_d a(a_rng);
        stan::math::matrix_d b(b_rng);
        const auto cpu_start = Clock::now();
        volatile auto c = multiply(a, b);
        const auto cpu_end = Clock::now();

        const auto gpu_start = Clock::now();
        volatile auto c_from_cl = multiply_cl(a, b);
        const auto gpu_end = Clock::now();
        
        experiments.emplace_back(InstanceSize {j, k, l},
            std::array<Duration,2> { cpu_end - cpu_start, gpu_end - gpu_start});
        
    }

    return experiments;
}

std::vector<Experiment> generate_cholesky_experiments(const std::vector<size_t> sizes) {
    std::vector<Experiment> experiments;

    for (auto j : sizes) {

        auto a_rng = stan::math::matrix_d::Random(j, j);
        stan::math::matrix_d a(a_rng);

        a = (a + a.transpose().eval()) / 2 + stan::math::matrix_d::Identity(j, j) * j;

        const auto cpu_start = Clock::now();
        auto c = cholesky_decompose(a);
        const auto cpu_end = Clock::now();

        const auto gpu_start = Clock::now();
        auto c_from_cl = cholesky_decompose_cl(a);
        const auto gpu_end = Clock::now();
        //std::cout << c(0) - c_from_cl(0) << "\n";
        
        experiments.emplace_back(InstanceSize {j, j},
            std::array<Duration,2> { cpu_end - cpu_start, gpu_end - gpu_start});
        
    }

    return experiments;
}

std::vector<Experiment> generate_sum_experiments(const std::vector<size_t> sizes) {
    std::vector<Experiment> experiments;

    for (auto j : sizes) {

        
        auto a_rng = stan::math::matrix_d::Random(j, 1);
        auto b_rng = stan::math::matrix_d::Random(j, 1);
        //std::cerr << "j = " << j << ", rep = " << 1 << std::endl;
        stan::math::matrix_d a(a_rng);
        stan::math::matrix_d b(b_rng);
        const auto cpu_start = Clock::now();
        volatile auto c = (a + b).eval();
        const auto cpu_end = Clock::now();

        const auto gpu_start = Clock::now();
        stan::math::matrix_cl<double> a_cl(a);
        stan::math::matrix_cl<double> b_cl(b);
        auto c_cl = (a_cl + b_cl).eval();
        volatile auto c_from_cl = stan::math::from_matrix_cl(c_cl).eval();
        const auto gpu_end = Clock::now();
        
        experiments.emplace_back(InstanceSize {j},
            std::array<Duration,2> { cpu_end - cpu_start, gpu_end - gpu_start});
        
    }

    return experiments;
}

#endif