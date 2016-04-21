# Invocation:
# gnuplot -e 'sim="simfile" trace="tracefile" orig="originalmodelfile" param=N' compare.plot 2> /dev/null
# For file output, add e.g. 'termtype="pdfcairo"; outfile="out.pdf";' to the parameters.

if (exists("termtype")) set terminal termtype
if (exists("outfile")) set output outfile

set table '/dev/null'
plot orig index 2*param+1 using 1:(CMD = sprintf('set obj rect from %f, graph 0 to %f, graph 1', $1, $2))
unset table
set style rect fc lt -1 fs solid 0.15 noborder
eval(CMD)

set title "Input waveform and response currents"
set ytics nomirror
set y2tics out
set ylabel "Current response (nA)"
set y2label "Input voltage (mV)"
set yrange [-1000 < *:* < 1000]

plot sim u 1:param+2 w lines t "Fitted model" axes x1y1, \
	orig index 2*param u 1:3 w lines t "Original model" axes x1y1, \
	trace index param u ($0/4.0):2 w lines t "Live neuron, median response" axes x1y1, \
	orig index 2*param u 1:2 w lines t "Voltage waveform" axes x1y2

pause -1
