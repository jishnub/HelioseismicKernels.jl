module HelioseismicKernels

using Reexport
@reexport using PointsOnASphere, JLD2, FileIO, WignerD, SphericalHarmonicModes

export Kernel3D, Kernel, Crosscov, Greenfn_radial, Directions, Traveltimes

import SphericalHarmonicArrays: shmodes
firstshmodes(x) = first(shmodes(x))

using ParallelUtilities
# Add a method to the finalizer to ignore `nothing`
ParallelUtilities.finalize_except_wherewhence(::Nothing) = nothing

@inline function Powspec(ω)
	σ = 2π*0.4e-3
	ω0 = 2π*3e-3
	exp(-(ω-ω0)^2/(2σ^2))
end

include("kernel3D.jl")
include("traveltimes.jl")
include("amplitudes.jl")
include("plots.jl")

end # module
