#include <random>

#include "strategy.hpp"
#include "result_utils.hpp"


int main(int argc, char *argv[]) {
    const size_t series = 20;
    //const size_t series = 1;
    // const size_t training_turns = 100;
    // const size_t turns = 100;
    //const size_t turns = 5;
    //const size_t series = 5;
    const size_t training_turns = 100;
    const size_t turns = 100;

    //const Problem problem = Problem::Multiplication;
    if (argc < 2) {
        throw std::logic_error("missing 'problem'");
    }
    Problem problem = (Problem)atoi(argv[1]);

    std::default_random_engine generator(0);
    std::vector<Experiment> training_experiments = get_training_experiments(problem, series, training_turns, generator);
    std::vector<Experiment> experiments;

    switch (problem) {
        case Problem::Multiplication: {
            std::uniform_int_distribution<size_t> distribution(150, 250);
            std::vector<size_t> experiment_sizes(series * turns);
            for (size_t i = 0; i < experiment_sizes.size(); i++) {
                experiment_sizes[i] = distribution(generator);
            }
            experiments = generate_experiments(experiment_sizes);
            break;
        }
        case Problem::CholeskyDecomposition: {
            std::uniform_int_distribution<size_t> distribution(275, 700);
            std::vector<size_t> experiment_sizes(series * turns);
            for (size_t i = 0; i < experiment_sizes.size(); i++) {
                experiment_sizes[i] = distribution(generator);
            }
            experiments = generate_cholesky_experiments(experiment_sizes);
            break;
        }
        case Problem::Summation: {
            std::uniform_int_distribution<size_t> distribution(275, 1'000'000);
            std::vector<size_t> experiment_sizes(series * turns);
            for (size_t i = 0; i < experiment_sizes.size(); i++) {
                experiment_sizes[i] = distribution(generator);
            }
            experiments = generate_sum_experiments(experiment_sizes);
            break;
        }
        default:
            throw std::out_of_range("problem out of range");
    }

    std::array<Eigen::Array<double, series, turns>, strategy_count> sum_time;

    for (size_t s = 0; s < strategy_count; s++) {
        uint correct = 0;
        for (size_t serie = 0; serie < series; serie++) {
            auto training_begin = training_experiments.begin() + (serie * training_turns);

            std::unique_ptr<Strategy> strategy;
            switch (problem) {
                case Problem::Multiplication:
                    strategy = std::unique_ptr<Strategy>(get_multiply_strategy((DecisionMethod)s,
                        training_begin,
                        training_begin + training_turns));
                    break;
                case Problem::CholeskyDecomposition:
                    strategy = std::unique_ptr<Strategy>(get_cholesky_strategy((DecisionMethod)s,
                        training_begin,
                        training_begin + training_turns));
                    break;
                case Problem::Summation:
                    strategy = std::unique_ptr<Strategy>(get_sum_strategy((DecisionMethod)s,
                        training_begin,
                        training_begin + training_turns));
                    break;
                default:
                    throw std::out_of_range("problem out of range");
            }

            for (size_t turn = 0; turn < turns; turn++) {
                auto experiment = experiments[serie * turns + turn];

                auto t0 = Clock::now();
                auto choice = strategy->choose(experiment.get_instance_size());
                strategy->update(experiment.get_instance_size(), choice, experiment.get_processing_time(choice));
                auto t1 = Clock::now();
                auto overhead = t1 - t0;
                //std::cerr << to_seconds(overhead) / to_seconds(experiment.get_processing_time(choice)) << " = " << to_seconds(overhead) << " / " << to_seconds(experiment.get_processing_time(choice)) << '\n';
                //overhead = Duration::zero(); //

                auto dt = to_seconds(experiment.get_processing_time(choice) + overhead);
                sum_time[s](serie, turn) = dt + (turn == 0 ? 0 : sum_time[s](serie, turn - 1));

                if (choice == (experiment.get_processing_time(CPU) < experiment.get_processing_time(GPU) ? CPU : GPU)) {
                    correct++;
                }
            }
        }
        std::cout << (double)correct / (series*turns) << ' ';
    }
    std::cout << '\n';
    
    Eigen::Array<double, series, turns> optimal_sum_time;
    for (size_t serie = 0; serie < series; serie++) {
        for (size_t turn = 0; turn < turns; turn++) {
            auto experiment = experiments[serie * turns + turn];
            auto processing_times = experiment.get_processing_time();
            auto dt = to_seconds(*std::min_element(processing_times.begin(), processing_times.end()));
            optimal_sum_time(serie, turn) = dt + (turn == 0 ? 0 : optimal_sum_time(serie, turn - 1));
        }
    }

    Eigen::Array<double, strategy_count, turns> relative_regret;
    for (size_t s = 0; s < strategy_count; s++) {
        for (size_t turn = 0; turn < turns; turn++) {
            relative_regret(s, turn) = (sum_time[s].col(turn) / optimal_sum_time.col(turn)).mean() - 1;
        }
    }

    std::ostream& output_stream = std::cout;

    //output_stream << sum_time[3].transpose() << "\n\n";
    //output_stream << optimal_sum_time.transpose() << "\n\n";
    output_stream << "#";
    for (auto name : strategy_names) {
        output_stream << name << '\t';
    }
    output_stream << '\n';
    output_stream << relative_regret.transpose() << '\n';


    return 0;
}