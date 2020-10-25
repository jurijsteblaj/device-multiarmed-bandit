#include <cmdstan/io/json/json_data.hpp>

#include "result_utils.hpp"
#include "strategy.hpp"
#include "models/GP_altered.hpp"
#include "result_utils.hpp"
#include "choice_funcs.hpp"
#include "timer_interrupt.hpp"

cmdstan::json::json_data get_context(const int file_ix) {
    std::ostringstream json_file_name;
    json_file_name << "data/input/dg_1d_nonlinear_" << file_ix << ".json";
    std::ifstream json_file(json_file_name.str());
    return cmdstan::json::json_data(json_file);
}

// GP_model_namespace::GP_model get_model(cmdstan::json::json_data gp_context, const size_t strategy_ix) {
//     auto multiply_func = get_choice_func<stan::math::matrix_d, const stan::math::matrix_d&, const stan::math::matrix_d&>({multiply, multiply_cl});
//     auto multiply_strategy = std::unique_ptr<Strategy>(get_multiply_strategy(strategy_ix));
//     auto simple_multiply_func = get_simple_choice_func(
//         multiply_func,
//         *multiply_strategy,
//         (std::function<InstanceSize(const stan::math::matrix_d&, const stan::math::matrix_d&)>)[](auto a, auto b) {
//             return InstanceSize { (size_t)a.rows(), (size_t)a.cols(), (size_t)b.cols() };
//         }
//     );
    
//     auto cholesky_func = get_choice_func<stan::math::matrix_d, const stan::math::matrix_d&>({cholesky_decompose, cholesky_decompose_cl});
//     auto cholesky_strategy = std::unique_ptr<Strategy>(get_multiply_strategy(strategy_ix)); // change to cholesky
//     auto simple_cholesky_func = get_simple_choice_func(
//         cholesky_func,
//         *cholesky_strategy,
//         (std::function<InstanceSize(const stan::math::matrix_d&)>)[](auto a) {
//             return InstanceSize { (size_t)a.rows(), (size_t)a.cols()};
//         }
//     );


//     return GP_model_namespace::GP_model(gp_context, simple_multiply_func, simple_cholesky_func);
// }

int main(int argc, char *argv[]) {
    //const size_t series = 5;
    //const size_t samples = 5;
    const size_t series = 20;
    const size_t samples = 20;
    //const size_t series = 1;
    //const size_t samples = 5;
    const size_t training_turns = 100;

    //int file_ix = 3;
    //double step = 1;
    //int file_ix = 18;
    //double step = ?;
    //int file_ix = 15; // 0.03
    //double step = 10;
    //int file_ix = 30; // 18
    //double step = 512.0;
    //int file_ix = 27; // 8
    if (argc < 2) {
        throw std::logic_error("missing 'file_ix'");
    }
    int file_ix = atoi(argv[1]);
    if (argc < 3) {
        throw std::logic_error("missing 'step'");
    }
    double step = atof(argv[2]);
    if (argc < 4) {
        throw std::logic_error("missing output 1");
    }
    if (argc < 5) {
        throw std::logic_error("missing output 2");
    }
    if (argc < 6) {
        throw std::logic_error("missing output 3");
    }
    

    auto gp_context = get_context(file_ix);

    unsigned int chain = 1;
    int num_thin = 1;
    bool save_warmup = false;
    int refresh = 0;
    double stepsize_jitter = 0.01;
    double int_time = 10;

    stan::test::unit::instrumented_logger logger;
    stan::test::unit::instrumented_writer init, parameter;
    stan::io::empty_var_context context;
    

    //std::array<DecisionMethod, 4> strategies { DecisionMethod::CPU, DecisionMethod::GPU, DecisionMethod::LinUCB, DecisionMethod::Normal };
    //std::array<DecisionMethod, 1> strategies { DecisionMethod::GPU };
    std::array<DecisionMethod, 8> strategies {
        DecisionMethod::CPU,
        DecisionMethod::GPU,
        DecisionMethod::LinUCB,
        DecisionMethod::Normal,
        DecisionMethod::Regression,
        DecisionMethod::TrainedLinUCB,
        DecisionMethod::TrainedNormal,
        DecisionMethod::TrainedRegression
    };

    std::default_random_engine generator(0);
    std::vector<Experiment> multiplication_training_experiments = get_training_experiments(Problem::Multiplication, series, training_turns, generator);
    std::vector<Experiment> cholesky_training_experiments = get_training_experiments(Problem::CholeskyDecomposition, series, training_turns, generator);

    std::array<Eigen::Array<double, series, samples>, strategies.size()> sample_times;

    Eigen::Array<double, strategies.size(), series> times;
    for (size_t s = 0; s < strategies.size(); s++) {
        for (size_t serie = 0; serie < series; serie++) {
            timer_writer diagnostic;
            timer_interrupt interrupt;

            auto training_begin = multiplication_training_experiments.begin() + (serie * training_turns);
            auto multiply_func = get_choice_func<stan::math::matrix_d, const stan::math::matrix_d&, const stan::math::matrix_d&>({multiply, multiply_cl});
            auto multiply_strategy = std::unique_ptr<Strategy>(get_multiply_strategy(strategies[s], training_begin, training_begin + training_turns));
            auto simple_multiply_func = get_simple_choice_func(
                multiply_func,
                *multiply_strategy,
                (std::function<InstanceSize(const stan::math::matrix_d&, const stan::math::matrix_d&)>)[](auto a, auto b) {
                    return InstanceSize { (size_t)a.rows(), (size_t)a.cols(), (size_t)b.cols() };
                }
            );
            
            training_begin = cholesky_training_experiments.begin() + (serie * training_turns);
            auto cholesky_func = get_choice_func<stan::math::matrix_d, const stan::math::matrix_d&>({cholesky_decompose, cholesky_decompose_cl});
            auto cholesky_strategy = std::unique_ptr<Strategy>(get_cholesky_strategy(strategies[s], training_begin, training_begin + training_turns));
            auto simple_cholesky_func = get_simple_choice_func(
                cholesky_func,
                *cholesky_strategy,
                (std::function<InstanceSize(const stan::math::matrix_d&)>)[](auto a) {
                    return InstanceSize { (size_t)a.rows(), (size_t)a.cols()};
                }
            );


            GP_model_namespace::GP_model model(gp_context, simple_multiply_func, simple_cholesky_func);

            auto t0 = Clock::now();
            auto status = stan::services::sample::hmc_nuts_dense_e(
                    model, context, serie, chain, 0, 0, samples,
                    num_thin, save_warmup, refresh, step, stepsize_jitter, int_time,
                    interrupt, logger, init, parameter, diagnostic);
            auto t1 = Clock::now();

            //std::cerr << diagnostic.call_count("vector_double") << '\n';
            for (size_t ti = 0; ti < interrupt.times().size(); ti++) {
                sample_times[s](serie, ti) = to_seconds(diagnostic.times(ti) - interrupt.times(ti));
            }
            //sample_times[s](serie, interrupt.times().size() - 1) = to_seconds(t1 - interrupt.times(interrupt.times().size() - 1));

            times(s, serie) = to_seconds(t1 - t0);
            std::cerr << status << ' ' << times(s, serie) << '\n';
        }
    }
    
    Eigen::Array<double, strategies.size(), samples> mean_sample_times;
    for (size_t s = 0; s < strategies.size(); s++) {
        for (size_t sample = 0; sample < samples; sample++) {
            mean_sample_times(s, sample) = sample_times[s].col(sample).mean();
        }
    }

    if (std::string(argv[3]) == "-") {
        std::cout << times.transpose() << '\n';
    }
    else {
        std::ofstream(argv[3])  << times.transpose() << '\n';
    }

    if (std::string(argv[4]) == "-") {
        std::cout << mean_sample_times.transpose() << '\n';
    }
    else {
        std::ofstream(argv[4])  << mean_sample_times.transpose() << '\n';
    }
    if (std::string(argv[5]) == "-") {
        for (auto matrix : sample_times) {
            std::cout << matrix << "\n\n";
        }
    }
    else {
        auto output_file = std::ofstream(argv[5]);
        for (auto matrix : sample_times) {
            output_file << matrix << "\n\n";
        }
    }

    // for (size_t s = 0; s < strategies.size(); s++) {
    //     for (size_t serie = 0; serie < series; serie++) {
    //         output_stream << times(s, serie) << "\t\"" << strategy_names[(size_t)strategies[s]] << "\"\n";
    //     }
    // }

    return 0;
}