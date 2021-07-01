#!/bin/bash
#SBATCH --time="1:00:00"
#SBATCH --job-name=kernel
#SBATCH -o kernel.out
#SBATCH -e kernel.err
#SBATCH --ntasks=56

cd $SCRATCH/jobs
# julia path, might not be necessary if provided by a module
julia="$PROJECT/julia-1.6.1/bin/julia"

$julia << EOF
module JobScript

using ClusterManagers, Distributed

addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]));

@everywhere begin
   using Pkg
   const HOME = ENV["HOME"]
   Pkg.activate("$HOME/HelioseismicKernels")
end

@time using HelioseismicKernels
using HelioseismicKernels: Point3D, r_obs_default

xobs1 = Point3D(r_obs_default, π/2, 0)
xobs2 = Point3D(r_obs_default, π/2, π/3)

s_min = 0
s_max = 10
ℓ_range = 1:100

HelioseismicKernels.kernel_uₛ₀_rθϕ(nothing,
   HelioseismicKernels.TravelTimes(), xobs1, xobs2; s_min = s_min, s_max = s_max,
   ℓ_range = 1:100, save = true)

end
EOF
