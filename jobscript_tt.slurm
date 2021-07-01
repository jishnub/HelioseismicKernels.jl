#!/bin/bash
#SBATCH --time="20"
#SBATCH --job-name=tt
#SBATCH -o traveltime.out
#SBATCH -e traveltime.err
#SBATCH --ntasks=56

cd $SCRATCH/jobs
# julia path, might not be necessary if provided by a module
julia="$PROJECT/julia-1.6.1/bin/julia"
module purge
module load openmpi
$julia -e 'include("$(ENV["HOME"])/HelioseismicKernels/compute_traveltimes.jl")'
