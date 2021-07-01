#!/bin/bash
#SBATCH --time="6:00:00"
#SBATCH --job-name=kernel
#SBATCH -o kernel.out
#SBATCH -e kernel.err
#SBATCH --ntasks=224

cd $SCRATCH/jobs
module purge
module load openmpi
julia="$PROJECT/julia-1.6.1/bin/julia"
mpirun $julia -e 'const HOME = ENV["HOME"]; include("$HOME/HelioseismicKernels/computekernel.jl")'
