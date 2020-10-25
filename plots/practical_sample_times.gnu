set terminal epslatex size 5.39,3.5 color colortext
set output output_file

set xlabel 'Vzorec'
set ylabel 'Čas'
plot input_file using 1 title 'CPE' with linesp lc rgb "black" pointtype 1,\
    input_file using 2 title 'GPE' with linesp lc rgb "gray" pointtype 1,\
    input_file using 3 title 'LinUCB' with lines lc rgb "red",\
    input_file using 6 title 'LinUCB*' with lines lc rgb "light-coral",\
    input_file using 4 title 'TL' with lines lc rgb "green",\
    input_file using 7 title 'TL*' with lines lc rgb "light-green",\
    input_file using 5 title 'TR' with lines lc rgb "blue",\
    input_file using 8 title 'TR*' with lines lc rgb "light-blue"