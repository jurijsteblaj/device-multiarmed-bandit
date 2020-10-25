set terminal epslatex size 5.39,3.5 color colortext
set output output_file

set style fill solid 0.25 border -1
set style boxplot outliers pointtype 7
set style data boxplot
#set xtics rotate by -45 ('CPE' 1, 'GPE' 2, 'LinUCB' 3, 'LinUCB*' 4, 'TL' 5, 'TL*' 6, 'TR' 7, 'TR*' 8)
set xtics rotate by -30 ('CPE' 1, 'GPE' 2, 'LinUCB' 3, 'LinUCB*' 4, 'TL' 5, 'TL*' 6, 'TR' 7, 'TR*' 8)
set nokey

#plot input_file using (1):1:(0.5):2
#plot 'data/gp-times.dat' using (1):1:(0.5):2
plot input_file using (1):1 title 'CPE' lc rgb "black",\
    '' using (2):2 title 'GPE' lc rgb "gray",\
    '' using (3):3 title 'LinUCB' lc rgb "red",\
    '' using (4):6 title 'LinUCB*' lc rgb "light-coral",\
    '' using (5):4 title 'TL' lc rgb "green",\
    '' using (6):7 title 'TL*' lc rgb "light-green",\
    '' using (7):5 title 'TR' lc rgb "blue",\
    '' using (8):8 title 'TR*' lc rgb "light-blue"