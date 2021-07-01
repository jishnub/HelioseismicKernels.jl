#!/bin/bash
#SBATCH --time="10"
#SBATCH --job-name=cc
#SBATCH -o crosscov.out
#SBATCH -e crosscov.err
#SBATCH --ntasks=56

cd $SCRATCH/jobs
module purge
module load openmpi
julia="$PROJECT/julia-1.6.1/bin/julia"
mpirun $julia -e 'const HOME = ENV["HOME"]; include("$HOME/HelioseismicKernels/computecrosscov.jl")'
