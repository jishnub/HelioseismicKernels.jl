module JobScript

using ClusterManagers, Distributed

addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]));

@everywhere begin
   using Pkg
   Pkg.activate("$(ENV["HOME"])/HelioseismicKernels")
end

using HelioseismicKernels

# angles are in degrees
ϕ_low = 35
ϕ_high = 65
nϕ = 10
ℓ_range = 1:100

_, _, D = HelioseismicKernels.δτ_Δϕ(nothing,
   HelioseismicKernels.TravelTimes(),
   HelioseismicKernels.Flow(), ϕ_low = ϕ_low, ϕ_high = ϕ_high, nϕ = nϕ,
   ℓ_range = ℓ_range, save = false)

show(stdout, MIME"text/plain"(), D)
println()

end
