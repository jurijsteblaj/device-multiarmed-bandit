#ifndef STRATEGY_HPP
#define STRATEGY_HPP

#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <stan/io/array_var_context.hpp>

#include "dataset.hpp"
#include "structures.hpp"

class Result {
private:
    InstanceSize _instance_size;
    Device _choice;
    TimeType _choice_time_start;
    TimeType _choice_time_end;
    Duration _processing_time;
public:
    Result(const InstanceSize instance_size, const Device choice, const TimeType time_start,
            const TimeType time_end, const Duration processing_time)
        : _instance_size(instance_size), _choice(choice), _choice_time_start(time_start),
        _choice_time_end(time_end), _processing_time(processing_time) {}
    
    auto get_instance_size() const { return _instance_size; }
    auto get_choice() const { return _choice; }
    auto get_time_start() const { return _choice_time_start; }
    auto get_time_end() const { return _choice_time_end; }
    auto get_choice_time() const {
        return _choice_time_end - _choice_time_start;
    }
    auto get_processing_time() const {
        return _processing_time;
    }
    auto get_total_time() const {
        return get_choice_time() + get_processing_time();
    }
};

/*
Assumptions:
1.  Order of experiments does not affect the runtime of a single experiment.
    This assumption is generally false because of instruction cache.

1.  Create a dataset - a vector of runtimes for both options.
2.  Randomize the order of the experiments.
3.  Feed the experiments one by one into a recorder.
*/
class Strategy {
protected:
    std::vector<Result> _results;

public:
    virtual Device choose(const InstanceSize instance_size) = 0;

    virtual void update(const InstanceSize instance_size, const Device choice, const Duration time) {
        // No learning.
    }

    void add_experiment(const Experiment experiment) {
        const auto choice_time_start = Clock::now();
        const Device choice = choose(experiment.get_instance_size());
        const auto choice_time_end = Clock::now();
        _results.emplace_back(experiment.get_instance_size(), choice,
            choice_time_start, choice_time_end, experiment.get_processing_time(choice));
        update(experiment.get_instance_size(), choice, experiment.get_processing_time(choice));
    }

    template<class Iterator>
    void add_experiments(const Iterator start, const Iterator end) {
        for (auto iter = start; iter != end; iter++) {
            add_experiment(*iter);
        }
    }

    auto get_choice_time() const {
        auto total = Duration::zero();
        for (auto experiment : _results) {
            total += experiment.get_choice_time();
        }
        return total;
    }

    auto get_total_time() const {
        auto total = Duration::zero();
        for (auto experiment : _results) {
            total += experiment.get_total_time();
        }
        return total;
    }

    auto results() const { return _results; }
};

template <Device device>
class AlwaysChoose : public Strategy {
public:
    Device choose(const InstanceSize instance_size) {
        return device;
    }
};

/*
P(CPU) = P(t(CPU) < t(GPU))
Actually P(t(CPU) < t(GPU)) / (P(t(CPU) < t(GPU)) + P(t(CPU) > t(GPU)))
but P(t(CPU) = t(GPU)) should be negligible.
*/
class Bandit : public Strategy {
private:
    std::default_random_engine generator;
public:
    Device choose(const InstanceSize instance_size) {
        std::array<size_t,2> victories {0, 0};
        for (auto i = 0; i < _results.size(); i++) {
            for (auto j = 0; j < _results.size(); j++) {
                if (_results[i].get_processing_time() < _results[j].get_processing_time()) {
                    victories[_results[i].get_choice()]++;
                }
            }
        }
        std::bernoulli_distribution distribution(static_cast<double>(victories[0]) / (victories[0] + victories[1]));
        return distribution(generator) ? CPU : GPU;
    }
};

class Bernoulli : public Strategy {
private:
    std::default_random_engine generator;
    std::bernoulli_distribution distribution;
public:
    Bernoulli(const double p = 0.5)
        : distribution(std::bernoulli_distribution(p)) {}

    Device choose(const InstanceSize instance_size) {
        return distribution(generator) ? CPU : GPU;
    }
};

class NormalBayes : public Strategy {
private:
    std::default_random_engine generator;
    std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> prior_mean_getter;
    std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> prior_var_getter;

    std::unordered_map<InstanceSize, std::array<std::vector<double>,DEVICE_COUNT>> processing_times;

    template <class ForwardIt>
    double mean(ForwardIt first, ForwardIt last, size_t n) {
        return std::accumulate(first, last, 0.0) / n;
    }

    template <class ForwardIt>
    double mean(ForwardIt first, ForwardIt last) {
        auto n = std::distance(first, last);
        return mean(first, last, n);
    }

    template <class ForwardIt>
    double sample_variance(ForwardIt first, ForwardIt last, double mean, size_t n) {
        auto sum = 0;
        for (auto it = first; it != last; it++) {
            auto diff = *it - mean;
            sum += diff*diff;
        }
        return sum / (n - 1);
    }
public:
    NormalBayes(std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> prior_mean_getter,
            std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> prior_var_getter)
        : prior_mean_getter(prior_mean_getter), prior_var_getter(prior_var_getter) {}

    Device choose(const InstanceSize instance_size) {
        std::array<double,DEVICE_COUNT> samples;
        for (auto device = 0; device < DEVICE_COUNT; device++) {
            const std::vector<double> processing_time = processing_times[instance_size][device];
            //std::cout << processing_time.size() << ' ';

            const auto prior_mean = prior_mean_getter[device](instance_size);
            const auto prior_var = prior_var_getter[device](instance_size);

            const auto n = processing_time.size();

            double posterior_mean;
            double posterior_var;
            if (n < 2) {
                posterior_mean = prior_mean;
                posterior_var = prior_var;
            }
            else {
                auto sample_mean = mean(processing_time.begin(), processing_time.end());
                auto sample_var = sample_variance(processing_time.begin(), processing_time.end(), sample_mean, n);
                posterior_mean = (sample_var*prior_mean + n*prior_var*sample_mean) /
                    (n * prior_var + sample_var);
                posterior_var = (sample_var * prior_var) /
                    (n * prior_var + sample_var);
            }
            
            std::normal_distribution<double> distribution(posterior_mean, std::sqrt(posterior_var));
            samples[device] = distribution(generator);
        }

        // std::cout << samples[0] << ' ' << samples[1] << ' ';
        Device max_ix = (Device)0;
        for (auto i = 1; i < DEVICE_COUNT; i++) {
            if (samples[max_ix] < samples[i]) {
                max_ix = (Device)i;
            }
        }

        // std::cout << max_ix;
        //std::cout << '\n';

        return max_ix;
    }

    void update(const InstanceSize instance_size, const Device choice, const Duration time) {
        processing_times[instance_size][choice].push_back(to_seconds(-time));
    }
};

class RegressionBayes : public Strategy {
private:
    std::default_random_engine generator;

    std::function<FeatureVector(InstanceSize)> feature_func;
    size_t feature_vector_size;
    
    std::array<FeatureVector, DEVICE_COUNT> Xs;
    std::array<std::vector<double>, DEVICE_COUNT> ys;

    stan::math::matrix_d Lambda_0;
    Eigen::VectorXd mu_0;
    double a_0;
    double b_0;

    double inverse_gamma(double shape, double scale) {
        return 1.0/std::gamma_distribution<double>(shape, 1.0/scale)(generator);
    }

public:
    RegressionBayes(const double precision, const std::function<FeatureVector(InstanceSize)> feature_func, const size_t feature_vector_size)
            : feature_func(feature_func), feature_vector_size(feature_vector_size) {
        Lambda_0 = stan::math::matrix_d::Identity(feature_vector_size, feature_vector_size) * precision;
        mu_0 = Eigen::VectorXd::Zero(feature_vector_size);
        a_0 = 0.01;
        b_0 = 0.01;
    }

    Device choose(const InstanceSize instance_size) {
        auto features = feature_func(instance_size);
        Eigen::Map<Eigen::VectorXd> feature_vec(features.data(), features.size());
        //std::cerr << "features: " << feature_vec << '\n';
        
        std::array<double,DEVICE_COUNT> samples;
        for (auto device = 0; device < DEVICE_COUNT; device++) {
            stan::math::matrix_d Lambda_n;
            Eigen::VectorXd mu_n;
            double a_n;
            double b_n;
            
            auto n = Xs[device].size() / feature_vector_size;
            if (n == 0) {
                Lambda_n = Lambda_0;
                mu_n = mu_0;
                a_n = a_0;
                b_n = b_0;
            }
            else {
                const Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> X(Xs[device].data(), Xs[device].size() / feature_vector_size, feature_vector_size);
                // for (auto v :  Xs[device]) {
                //     std::cerr << v << ' ';
                // }
                //std::cerr << '\n';
                //std::cerr << "design: " << X << '\n';
                const Eigen::Map<Eigen::VectorXd> y(ys[device].data(), ys[device].size());
                
                Lambda_n = X.transpose()*X + Lambda_0;
                mu_n = Lambda_n.inverse() * (Lambda_0 * mu_0 + X.transpose() * y);
                a_n = a_0 + n / 2.0;
                b_n = b_0 + (double)(y.dot(y) + mu_0.transpose()*Lambda_0*mu_0 - mu_n.transpose()*Lambda_n*mu_n)/2.0;
            }
            auto var = inverse_gamma(a_n, b_n);

            //std::cerr << Lambda_n << '\n';
            //std::cerr << Lambda_n.inverse() << '\n' << std::endl;
            
            // std::cerr << "device: " << device << '\n';
            // std::cerr << "a: " << a_n << ", b: " << b_n << '\n';
            // std::cerr << "var: " << var << '\n';
            // std::cerr << "Lambda_n = " << Lambda_n << '\n';
            // std::cerr << "covar: " << Eigen::MatrixXd(var*Lambda_n.inverse().selfadjointView<Eigen::Upper>()) << '\n';
            // std::cerr << mu_n << '\n';
            // std::cerr << "expected: " << mu_n.dot(feature_vec) << '\n';

            auto b = stan::math::multi_normal_rng(mu_n, var*Lambda_n.inverse().selfadjointView<Eigen::Upper>(), generator);
            samples[device] = feature_vec.dot(b);
            // std::cerr << "sample: " << samples[device] << '\n';
        }

        Device max_ix = (Device)0;
        for (auto i = 1; i < DEVICE_COUNT; i++) {
            if (samples[max_ix] < samples[i]) {
                max_ix = (Device)i;
            }
        }

        return max_ix;
    }

    void update(const InstanceSize instance_size, const Device choice, const Duration time) {
        auto feature_vector = feature_func(instance_size);
        Xs[choice].insert(Xs[choice].end(), feature_vector.begin(), feature_vector.end());
        ys[choice].push_back(to_seconds(-time));
    }
};

class DisjointLinUCB : public Strategy {
private:
    double alpha;
    std::function<FeatureVector(InstanceSize)> feature_func;

    std::array<Eigen::Matrix<double, -1, -1>, DEVICE_COUNT> A;
    std::array<Eigen::Matrix<double, -1, 1>, DEVICE_COUNT> b;
public:
    DisjointLinUCB(const double alpha, const std::function<FeatureVector(InstanceSize)> feature_func, const size_t feature_vector_size)
            : alpha(alpha), feature_func(feature_func) {
        const auto d = feature_vector_size;
        for (auto device = 0; device < DEVICE_COUNT; device++) {
            A[device] = Eigen::Matrix<double, -1, -1>::Identity(d, d);
            b[device] = Eigen::Matrix<double, -1, 1>::Zero(d);
        }
    }

    Device choose(const InstanceSize instance_size) {
        std::array<double, DEVICE_COUNT> p;
        // Single feature vector for all arms.
        auto features = feature_func(instance_size);
        Eigen::Map<Eigen::Matrix<double, -1, 1>> x(features.data(), features.size());

        for (auto device = 0; device < DEVICE_COUNT; device++) {
            auto theta_transposed = (A[device].inverse() * b[device]).transpose().eval();
            p[device] = theta_transposed*x + alpha*sqrt(x.transpose()*A[device].inverse()*x);
        }

        Device max_ix = (Device)0;
        for (auto i = 1; i < DEVICE_COUNT; i++) {
            if (p[max_ix] < p[i]) {
                max_ix = (Device)i;
            }
        }

        return max_ix;
    }

    void update(const InstanceSize instance_size, const Device choice, const Duration time) {
        auto features = feature_func(instance_size);
        Eigen::Map<Eigen::Matrix<double, -1, 1>> x(features.data(), features.size());
        A[choice] += x*x.transpose();
        b[choice] += to_seconds(-time) * x;
    }
};

/*
1. load model (parameters or samples)
2. posterior_mean = beta * instance_size
3. update model
*/

/*
Pretraining - small instance:
1. Linear regression of t_device ~ f(instance_size) on precomputed dataset.

Online learning:
1. As in NormalBayes, using results of regressions as priors.

*/

#endif