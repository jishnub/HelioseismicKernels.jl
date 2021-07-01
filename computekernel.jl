using Pkg
const HOME = ENV["HOME"]
using Printf
Pkg.activate("$HOME/HelioseismicKernels")

using MPI
MPI.Init()
using HelioseismicKernels
const comm = MPI.COMM_WORLD

const xobs1 = Point2D(pi/2, pi/3)
const xobs2 = Point2D(pi/2, pi/2)
const ℓ_range = 1:100

try
    HelioseismicKernels.kernel_uₛₜ(comm, TravelTimes(), xobs1, xobs2, los_radial(), save = false, ℓ_range = 1:1)
    HelioseismicKernels.kernel_uₛₜ(comm, TravelTimes(), xobs1, xobs2, los_earth(), save = false, ℓ_range = 1:1)

    s_max = 25

    tstart = time()
    HelioseismicKernels.kernel_uₛₜ(comm,
        TravelTimes(), xobs1, xobs2, los_earth(),
        s_max = s_max, t_max = s_max, ℓ_range = ℓ_range)

    tend = time()
    Δt = round(Int, tend - tstart)
    if MPI.Comm_rank(comm) == 0
        @printf "los\t%d\t%d\n" s_max Δt
    end

    tstart = time()

    HelioseismicKernels.kernel_uₛₜ(comm,
        TravelTimes(), xobs1, xobs2, los_radial(),
        s_max = s_max, t_max = s_max, ℓ_range = ℓ_range)

    tend = time()
    Δt = round(Int, tend - tstart)
    if MPI.Comm_rank(comm) == 0
        @printf "rad\t%d\t%d\n" s_max Δt
    end

finally
    MPI.Finalize()
end
