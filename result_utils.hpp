#ifndef RESULT_UTILS_HPP
#define RESULT_UTILS_HPP

#include "prior_generators.hpp"
#include "strategy.hpp"
#include "feature_vector.hpp"

const size_t strategy_count = 8;

enum class DecisionMethod {
    CPU,
    GPU,
    LinUCB,
    Normal,
    Regression,
    TrainedLinUCB,
    TrainedNormal,
    TrainedRegression
};

enum class Problem {
    Multiplication,
    CholeskyDecomposition,
    Summation
};

std::array<std::string, strategy_count> strategy_names { "CPU", "GPU", "LinUCB", "Normal", "Regression", "Trained LinUCB", "Trained Normal", "Trained Regression" };

namespace result::config {
    // const auto feature_func = [](InstanceSize instance_size) {
    //     return get_polynomial_feature_vector(instance_size, 1);
    // };
    // const size_t feature_vector_size = 8;

    // const auto feature_func = [](InstanceSize instance_size) {
    //     return std::vector<double> { 1, (double)instance_size[0], (double)instance_size[1], (double)instance_size[2] };
    // };
    // const size_t feature_vector_size = 4;

    // const auto feature_func = [](InstanceSize instance_size) {
    //     return std::vector<double> { 1 };
    // };
    // const size_t feature_vector_size = 1;

    // FEATURE FUNCTIONS NEED TO BE PROBLEM-SPECIFIC BECAUSE OF NUMBER OF PARAMETER

    const double alpha = 0.00001;

    const double mean = 0;
    const double var = 100;

    const double precision = 1e3;
}

template <typename Iterator>
void train_strategy(Strategy* strategy, Iterator begin, const Iterator end) {
    for (; begin < end; begin++) {
        auto experiment = *begin;
        for (auto i = 0; i < DEVICE_COUNT; i++) {
            strategy->update(experiment.get_instance_size(), (Device)i, experiment.get_processing_time((Device)i));
        }
    }
}

template <typename Iterator>
Strategy* get_multiply_strategy(const DecisionMethod strategy_ix, Iterator begin, const Iterator end) {
    const auto feature_func = [](InstanceSize instance_size) {
        return std::vector<double> { 1, (double)instance_size[0], (double)instance_size[1], (double)instance_size[2] };
    };
    const size_t feature_vector_size = 4;
    // const auto feature_func = [](InstanceSize instance_size) {
    //     return get_polynomial_feature_vector(instance_size, 1);
    // };
    // const size_t feature_vector_size = 8;

    switch (strategy_ix) {
        case DecisionMethod::CPU:
            return new AlwaysChoose<CPU>();
        case DecisionMethod::GPU:
            return new AlwaysChoose<GPU>();
        case DecisionMethod::LinUCB:
            return new DisjointLinUCB(result::config::alpha, feature_func, feature_vector_size);
        case DecisionMethod::Normal:
            return new NormalBayes(
                std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> { get_func(result::config::mean), get_func(result::config::mean) },
                std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> { get_func(result::config::var), get_func(result::config::var) });
        case DecisionMethod::Regression:
            return new RegressionBayes(result::config::precision, feature_func, feature_vector_size);
        case DecisionMethod::TrainedLinUCB: {
            auto strategy = new DisjointLinUCB(result::config::alpha, feature_func, feature_vector_size);
            train_strategy(strategy, begin, end);
            return strategy;
        }
        case DecisionMethod::TrainedNormal: {
            arma::Mat<double> instance_sizes(end - begin, feature_vector_size);
            arma::Mat<double> times(end - begin, DEVICE_COUNT);
            for (auto i = 0; begin + i < end; i++) {
                auto experiment = *(begin + i);
                instance_sizes.row(i) = arma::conv_to<arma::Row<double>>::from(feature_func(experiment.get_instance_size()));
                for (auto j = 0; j < DEVICE_COUNT; j++) {
                    times(i,j) = to_seconds(experiment.get_processing_time()[j]);
                }
            }

            std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> mean_funcs;
            std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> var_funcs;
            for (auto i = 0; i < DEVICE_COUNT; i++) {
                mlpack::regression::LinearRegression lr(instance_sizes.t(), times.col((Device)i).t(), 0, false);
                mean_funcs[i] = get_func(lr, feature_func);
                var_funcs[i] = get_var_func(lr, instance_sizes.t(), times.col((Device)i).t(), feature_func);
            }

            auto strategy = new NormalBayes(
                mean_funcs,
                var_funcs);
            return strategy;
        }
        case DecisionMethod::TrainedRegression: {
            auto strategy = new RegressionBayes(result::config::precision, feature_func, feature_vector_size);
            train_strategy(strategy, begin, end);
            return strategy;
        }
        default:
            throw std::out_of_range("strategy_ix out of range");
    }
}

template <typename Iterator>
Strategy* get_cholesky_strategy(const DecisionMethod strategy_ix, Iterator begin, const Iterator end) {
    const auto feature_func = [](InstanceSize instance_size) {
        return std::vector<double> { 1, (double)instance_size[0], (double)instance_size[1] };
    };
    const size_t feature_vector_size = 3;

    switch (strategy_ix) {
        case DecisionMethod::CPU:
            return new AlwaysChoose<CPU>();
        case DecisionMethod::GPU:
            return new AlwaysChoose<GPU>();
        case DecisionMethod::LinUCB:
            return new DisjointLinUCB(result::config::alpha, feature_func, feature_vector_size);
        case DecisionMethod::Normal:
            return new NormalBayes(
                std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> { get_func(result::config::mean), get_func(result::config::mean) },
                std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> { get_func(result::config::var), get_func(result::config::var) });
        case DecisionMethod::Regression:
            return new RegressionBayes(result::config::precision, feature_func, feature_vector_size);
        case DecisionMethod::TrainedLinUCB: {
            auto strategy = new DisjointLinUCB(result::config::alpha, feature_func, feature_vector_size);
            train_strategy(strategy, begin, end);
            return strategy;
        }
        case DecisionMethod::TrainedNormal: {
            arma::Mat<double> instance_sizes(end - begin, feature_vector_size);
            arma::Mat<double> times(end - begin, DEVICE_COUNT);
            for (auto i = 0; begin + i < end; i++) {
                auto experiment = *(begin + i);
                instance_sizes.row(i) = arma::conv_to<arma::Row<double>>::from(feature_func(experiment.get_instance_size()));
                for (auto j = 0; j < DEVICE_COUNT; j++) {
                    times(i,j) = to_seconds(experiment.get_processing_time()[j]);
                }
            }

            std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> mean_funcs;
            std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> var_funcs;
            for (auto i = 0; i < DEVICE_COUNT; i++) {
                mlpack::regression::LinearRegression lr(instance_sizes.t(), times.col((Device)i).t(), 0, false);
                mean_funcs[i] = get_func(lr, feature_func);
                var_funcs[i] = get_var_func(lr, instance_sizes.t(), times.col((Device)i).t(), feature_func);
            }

            auto strategy = new NormalBayes(
                mean_funcs,
                var_funcs);
            return strategy;
        }
        case DecisionMethod::TrainedRegression: {
            auto strategy = new RegressionBayes(result::config::precision, feature_func, feature_vector_size);
            train_strategy(strategy, begin, end);
            return strategy;
        }
        default:
            throw std::out_of_range("strategy_ix out of range");
    }
}

template <typename Iterator>
Strategy* get_sum_strategy(const DecisionMethod strategy_ix, Iterator begin, const Iterator end) {
    const auto feature_func = [](InstanceSize instance_size) {
        return std::vector<double> { 1, (double)instance_size[0] };
    };
    const size_t feature_vector_size = 2;

    switch (strategy_ix) {
        case DecisionMethod::CPU:
            return new AlwaysChoose<CPU>();
        case DecisionMethod::GPU:
            return new AlwaysChoose<GPU>();
        case DecisionMethod::LinUCB:
            return new DisjointLinUCB(result::config::alpha, feature_func, feature_vector_size);
        case DecisionMethod::Normal:
            return new NormalBayes(
                std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> { get_func(result::config::mean), get_func(result::config::mean) },
                std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> { get_func(result::config::var), get_func(result::config::var) });
        case DecisionMethod::Regression:
            return new RegressionBayes(result::config::precision, feature_func, feature_vector_size);
        case DecisionMethod::TrainedLinUCB: {
            auto strategy = new DisjointLinUCB(result::config::alpha, feature_func, feature_vector_size);
            train_strategy(strategy, begin, end);
            return strategy;
        }
        case DecisionMethod::TrainedNormal: {
            arma::Mat<double> instance_sizes(end - begin, feature_vector_size);
            arma::Mat<double> times(end - begin, DEVICE_COUNT);
            for (auto i = 0; begin + i < end; i++) {
                auto experiment = *(begin + i);
                instance_sizes.row(i) = arma::conv_to<arma::Row<double>>::from(feature_func(experiment.get_instance_size()));
                for (auto j = 0; j < DEVICE_COUNT; j++) {
                    times(i,j) = to_seconds(experiment.get_processing_time()[j]);
                }
            }

            std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> mean_funcs;
            std::array<std::function<double(InstanceSize)>,DEVICE_COUNT> var_funcs;
            for (auto i = 0; i < DEVICE_COUNT; i++) {
                mlpack::regression::LinearRegression lr(instance_sizes.t(), times.col((Device)i).t(), 0, false);
                mean_funcs[i] = get_func(lr, feature_func);
                var_funcs[i] = get_var_func(lr, instance_sizes.t(), times.col((Device)i).t(), feature_func);
            }

            auto strategy = new NormalBayes(
                mean_funcs,
                var_funcs);
            return strategy;
        }
        case DecisionMethod::TrainedRegression: {
            auto strategy = new RegressionBayes(result::config::precision, feature_func, feature_vector_size);
            train_strategy(strategy, begin, end);
            return strategy;
        }
        default:
            throw std::out_of_range("strategy_ix out of range");
    }
}

template <class Generator>
auto get_training_experiments(const Problem problem, const size_t series, const size_t training_turns, Generator& generator) {
    switch (problem) {
        case Problem::Multiplication: {
            std::uniform_int_distribution<size_t> training_distribution(1, 200);
            std::vector<size_t> training_experiment_sizes(series * training_turns);
            for (size_t i = 0; i < training_experiment_sizes.size(); i++) {
                training_experiment_sizes[i] = training_distribution(generator);
            }
            return generate_experiments(training_experiment_sizes);
        }
        case Problem::CholeskyDecomposition: {
            std::uniform_int_distribution<size_t> training_distribution(1, 200); //nastavi obe distribuciji in mogoče priorje
            std::vector<size_t> training_experiment_sizes(series * training_turns);
            for (size_t i = 0; i < training_experiment_sizes.size(); i++) {
                training_experiment_sizes[i] = training_distribution(generator);
            }
            return generate_cholesky_experiments(training_experiment_sizes);
        }
        case Problem::Summation: {
            std::uniform_int_distribution<size_t> training_distribution(1, 200);
            std::vector<size_t> training_experiment_sizes(series * training_turns);
            for (size_t i = 0; i < training_experiment_sizes.size(); i++) {
                training_experiment_sizes[i] = training_distribution(generator);
            }
            return generate_sum_experiments(training_experiment_sizes);
        }
        default:
            throw std::out_of_range("problem out of range");
    }
}

#endif