#!/usr/bin/env bash
set -e

for params in "3 1" "15 10" "27 512"; do
    suffix=""
    for param in $params; do
        suffix="$(echo $suffix)_$param"
    done

    data_file1="data/practical_results$suffix.dat"
    data_file2="data/practical_sample_times$suffix.dat"
    ./generate_practical_results $params $data_file1 $data_file2 "data/practical_all_sample_times$suffix.dat"
    
    gnuplot -e "input_file='$data_file1'" \
        -e "output_file='plots/practical_results$suffix.tex'" plots/practical_results.gnu
    gnuplot -e "input_file='$data_file2'" \
        -e "output_file='plots/practical_sample_times$suffix.tex'" plots/practical_sample_times.gnu
    echo "Finished params '$params' at $(date)"
done