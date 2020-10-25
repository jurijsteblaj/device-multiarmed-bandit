#!/usr/bin/env bash
set -e

for problem in 0 1 2; do
    data_file="data/method_results_$problem.dat"
    ./generate_method_results $problem > $data_file

    gnuplot -e "input_file='$data_file'" \
        -e "output_file='plots/method_results_$problem.tex'" plots/method_results.gnu
    echo "Finished problem $problem at $(date)"
done