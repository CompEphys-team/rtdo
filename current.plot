# Invocation:
# Default (x11) terminal:
# gnuplot -e 'file="filename"; param=N' current.plot 2> /dev/null
# Output as figure:
# gnuplot -e 'file="filename"; param=N; termtype="fig"; outfile="outfile.fig"' current.plot 2> /dev/null

if (!exists("file")) exit
if (!exists("param")) param = 0
if (exists("termtype")) set terminal termtype
if (exists("outfile")) set output outfile

set table '/dev/null'
plot file index 2*param+1 using 1:(CMD = sprintf('set obj rect from %f, graph 0 to %f, graph 1', $1, $2))
unset table

set style rect fc lt -1 fs solid 0.15 noborder
eval(CMD)

set key autotitle columnhead
set key below
set title "Current response for reference and detuned models, target parameter ".param
plot for [idx=2:50] file index 2*param using 1:idx with lines lw (idx==3 || idx==param+4 ? 3 : 1)

pause -1

