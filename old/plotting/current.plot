# Invocation:
# Default (x11) terminal:
# gnuplot -e 'file="filename"; param=0; first=0; last=10' current.plot 2> /dev/null
# gnuplot -e 'file="filename"; param=0; params="0 1 2 3 4"' current.plot 2> /dev/null
# Output as figure:
# gnuplot -e 'file="filename"; param=N; termtype="fig"; outfile="outfile.fig"' current.plot 2> /dev/null
#
# Defaults: param=0, first=0, last=10
# If params is set (to a string of whitespace-separated param numbers), this will be used instead of first and last.

if (!exists("file")) exit
if (!exists("param")) param = 0
if (!exists("first")) first = 0
if (!exists("last")) last = 10
if (!exists("params")) { params = ""; useParamList = 0 } else { useParamList = 1 }
if (last < first) last = first
if (exists("termtype")) set terminal termtype
if (exists("outfile")) set output outfile

set table '/dev/null'
plot file index 2*param+1 using 1:(CMD = sprintf('set obj rect from %f, graph 0 to %f, graph 1', $1, $2))
unset table

set style rect fc lt -1 fs solid 0.15 noborder
eval(CMD)

set key autotitle columnhead
set xtics mirror out
set ytics out
set y2tics out

set multiplot title "Waveform and current response, target parameter ".param layout 2,1

set title "Input waveform and reference current"
set ytics nomirror
set ylabel "Input voltage (mV)"
set y2label "Current response (nA)"
plot file index 2*param using 1:2 with lines axes x1y1, file index 2*param using 1:3 with lines axes x1y2

set key center rmargin vertical maxcols 1 samplen 2 spacing 1
set title "Absolute current difference from detuned models to reference"
set ylabel "|reference - detuned| (nA)"
unset y2label
unset y2tics
set ytics mirror
if (useParamList == 0) {
	plot for [idx=first:last] file index 2*param using 1:idx+4 with lines lw (idx==param ? 2 : 1)
} else {
	plot for [n=1:words(params)]  file index 2*param using 1:(column(word(params,n)+4)) \
		with lines lw (word(params,n)==param ? 2 : 1) \
		title columnhead(word(params,n)+4)
}

unset multiplot

pause -1

