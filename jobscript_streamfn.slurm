#!/bin/bash
#SBATCH --time="1:00:00"
#SBATCH --job-name=ttstrmfn
#SBATCH -o streamfn.out
#SBATCH -e streamfn.err
#SBATCH --ntasks=56

cd $SCRATCH/jobs
# julia path, might not be necessary if provided by a module
julia="$PROJECT/julia-1.6.1/bin/julia"
module purge
module load openmpi
$julia -e 'include("$(ENV["HOME"])/HelioseismicKernels/streamfn_traveltimes.jl")'
