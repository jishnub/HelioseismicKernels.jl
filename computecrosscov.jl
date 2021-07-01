using Pkg
using Dates
const HOME = ENV["HOME"]
Pkg.activate("$HOME/HelioseismicKernels")

using MPI
MPI.Init()
using HelioseismicKernels
const comm = MPI.COMM_WORLD
const nworkers = MPI.Comm_size(comm)

try
    HelioseismicKernels.Cω(comm, Point2D(π/2, 0), Point2D(π/2, deg2rad(45)), los_earth(),
    ν_ind_range = 1:nworkers)

    tstart = time()
    HelioseismicKernels.Cω(comm, Point2D(π/2, 0), Point2D(π/2, deg2rad(45)), los_earth())
    tend = time()
    Δt = Time(0) + Second(round(Int, tend - tstart))
    if MPI.Comm_rank(comm) == 0
        println("Time taken = $Δt")
    end
finally
    MPI.Finalize()
end
