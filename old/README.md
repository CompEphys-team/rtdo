# rtdo
Real time dynamical observer


For Voltage clamp:
1) Get all the env variabes going (replace with your paths):

export GENN_PATH=/mnt/nfs2/inf/ds376/genn

export PATH=/mnt/nfs2/inf/ds376/genn/lib/bin:$PATH

export CUDA_PATH=/cm/shared/apps/cuda/6.5/



2)Compile generator/runner (assuming you are in vclamp directory):

make



3) Create output directory:

mkdir wave_output



4) Because of some sussex HPC relate issues, generating and running is split to two different binaries:

./generate 1 0 1000 200000 wave wave2.dat sigma2.dat -1 0 0

./run 1 0 1000 200000 wave wave2.dat sigma2.dat -1 0 0

If running in RT mode with a live neuron hooked up, use instead:

./run 2 0 1000 200000 wave wave2.dat sigma2.dat -1 0 0


'Real neuron' method is in helper.h:

void runexpHH( float t )
