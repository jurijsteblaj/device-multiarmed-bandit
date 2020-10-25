#ifndef PRIOR_GENERATORS_HPP
#define PRIOR_GENERATORS_HPP

#include <functional>

#include <stan/io/empty_var_context.hpp>
#include <stan/math.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/sample/fixed_param.hpp>
#include <stan/services/sample/hmc_nuts_dense_e.hpp>
#include <stan/services/sample/standalone_gqs.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

#include "structures.hpp"
#include "lib/instrumented_callbacks.hpp"

std::function<double(InstanceSize)> get_func(
        mlpack::regression::LinearRegression lr,
        std::function<FeatureVector(InstanceSize)> feature_func) {
    return [lr, feature_func](InstanceSize instance_size) {
        arma::Row<double> predictions(1);
        auto feature_vector = feature_func(instance_size);
        lr.Predict(arma::conv_to<arma::Col<double>>::from(feature_vector), predictions);
        return predictions[0];
    };
}

std::function<double(InstanceSize)> get_var_func(
        mlpack::regression::LinearRegression& lr,
        const arma::mat& points,
        const arma::rowvec& responses) {
    double var = lr.ComputeError(points, responses);
    return [var](InstanceSize instance_size) {
        return var;
    };
}

std::function<double(InstanceSize)> get_var_func(
        mlpack::regression::LinearRegression& lr,
        const arma::mat& points,
        const arma::rowvec& responses,
        const std::function<FeatureVector(InstanceSize)>& feature_func) {

    arma::rowvec predictions(responses.size());
    lr.Predict(points, predictions);

    mlpack::regression::LinearRegression lr_var(points, arma::pow(responses - predictions, 2), 0, false);
    return [lr_var, feature_func](InstanceSize instance_size) {
        arma::Row<double> predictions(1);
        auto feature_vector = feature_func(instance_size);
        lr_var.Predict(arma::conv_to<arma::Col<double>>::from(feature_vector), predictions);
        return predictions[0];
    };
}

// std::function<double(InstanceSize)> get_func(
//         std::function<stan::io::array_var_context(const std::vector<double>&)>& var_context_constructor,
//         Eigen::Matrix<double, -1, -1>& parameter_samples,
//         std::default_random_engine& generator,
//         std::function<FeatureVector(InstanceSize)>& feature_func) {

//     return [var_context_constructor, parameter_samples, feature_func, &generator](InstanceSize instance_size) {
//         stan::test::unit::instrumented_logger pred_logger;
//         stan::test::unit::instrumented_writer pred_parameter;
//         stan::test::unit::instrumented_interrupt pred_interrupt;
//         std::uniform_int_distribution<size_t> distribution(0, parameter_samples.rows() - 1);

//         auto pred_context = var_context_constructor(feature_func(instance_size));
//         StanModel pred_model(pred_context);
//         auto ix = distribution(generator);
//         stan::services::standalone_generate(pred_model, parameter_samples.row(ix), 0, pred_interrupt, pred_logger, pred_parameter);

//         return pred_parameter.vector_double_values()[0][0];
//     };
// }

std::function<double(InstanceSize)> get_var_func(
        Eigen::Matrix<double, -1, -1>& parameter_samples,
        std::default_random_engine& generator) {

    return [parameter_samples, &generator](InstanceSize instance_size) {
        std::uniform_int_distribution<size_t> distribution(0, parameter_samples.rows() - 1);
        auto ix = distribution(generator);
        auto sigma = parameter_samples(ix, parameter_samples.cols() - 1);
        return sigma*sigma;
    };
}

std::function<double(InstanceSize)> get_func(double number) {
    return [number](InstanceSize instance_size) {
        return number;
    };
}

#endif