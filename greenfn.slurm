#!/bin/bash
#SBATCH --time="00:30:00"
#SBATCH --job-name=gfn
#SBATCH -o greenfn.out
#SBATCH -e greenfn.err
#SBATCH --ntasks=420

cd $SCRATCH/jobs
julia="$PROJECT/julia-1.6.1/bin/julia"
module purge
module load openmpi

$julia << EOF

using ClusterManagers, Distributed

# Make the slurm cores available to Julia
addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]));

# This might help with loading packages, not too sure.
# It's not necessary as such for the code
@everywhere pushfirst!(Base.DEPOT_PATH, "/tmp/julia.cache")

@everywhere begin
   using Pkg
   const HOME = ENV["HOME"]
   Pkg.activate("$HOME/HelioseismicKernels")
end

@time using HelioseismicKernels

const num_ν = 4000
const ℓ_arr = 1:100
const r_obs = HelioseismicKernels.r_obs_default
const r_src = HelioseismicKernels.r_src_default

@time HelioseismicKernels.greenfn_components(r_src, num_ν = num_ν, ℓ_arr = ℓ_arr);
@time HelioseismicKernels.greenfn_components(r_src, num_ν = num_ν, ℓ_arr = ℓ_arr, c_scale = 1 + 1e-5);
@time HelioseismicKernels.greenfn_components(r_obs, num_ν = num_ν, ℓ_arr = ℓ_arr);

EOF
