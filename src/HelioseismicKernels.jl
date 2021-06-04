module HelioseismicKernels

using BipolarSphericalHarmonics
using BipolarSphericalHarmonics: kronindex
using DSP: hilbert
using DelimitedFiles
using Distributed
using DualNumbers
using EllipsisNotation
using FastGaussQuadrature
using FFTW
using FileIO
using FITSIO
using HybridArrays
using JLD2
using LegendrePolynomials
using LinearAlgebra
BLAS.set_num_threads(1)
using LoopVectorization
using MPI
using NamedArrays
using NumericalIntegration
using OffsetArrays
using OffsetArrays: no_offset_view
using ParallelUtilities
using ParallelUtilities: ProductSplit, extremaelement
using Polynomials
using Printf
using ProgressMeter
using Rotations
using SparseArrays
using SphericalHarmonicModes
using SphericalHarmonicArrays
using SphericalHarmonicArrays: shmodes
using SphericalHarmonics
using StaticArrays
using StructArrays
using SuiteSparse
using TimerOutputs
using UnPack
using DistributedArrays
using VectorSphericalHarmonics
using WignerD
using WignerSymbols

export TravelTimes, Amplitudes, SoundSpeed, Flow
export Point2D, Point3D
export los_radial, los_earth

firstshmodes(x) = first(shmodes(x))

# remove wrappers to get the underlying array
reinterpret_as_float(arr::OffsetArray) = reinterpret_as_float(no_offset_view(arr))
# convert complex arrays to real
reinterpret_as_float(arr::Array{<:Complex}) = reinterpret(real(eltype(arr)), arr)
# real arrays may be returned as is
reinterpret_as_float(arr::Array{<:Real}) = arr

reinterpret_as_complex(arr::OffsetArray) = reinterpret_as_complex(no_offset_view(arr))
reinterpret_as_complex(arr::Array{<:Complex}) = arr
reinterpret_as_complex(arr::Array{<:Real}) = reinterpret(complex(eltype(arr)), arr)

function save_to_fits_and_return(filename, arr::Array; kwargs...)
	filepath = joinpath(SCRATCH_kerneldir[], filename)
	FITS(filepath, "w") do f
		outarr = reinterpret_as_float(arr)
		FITSIO.write(f, outarr; kwargs...)
	end
	arr
end

function save_to_fits_and_return(filename, val::Number; kwargs...)
	save_to_fits_and_return(filename, [val]; kwargs...)
end

function Powspec(ω)
	σ = 2π*0.4e-3
	ω0 = 2π*3e-3
	exp(-(ω-ω0)^2/(2σ^2))
end

#################################################################
# Cross covariances and changes in cross covariances
#################################################################
abstract type SeismicMeasurement end
struct TravelTimes <: SeismicMeasurement end
struct Amplitudes <: SeismicMeasurement end

abstract type PerturbationParameter end
struct SoundSpeed <: PerturbationParameter end
struct Flow <: PerturbationParameter end

##########################################################################################

const SCRATCH = Ref("")
const SCRATCH_kerneldir = Ref("")

function __init__()
    SCRATCH[] = get(ENV, "SCRATCH",
                isdir(joinpath("/scratch", ENV["USER"])) ?
                joinpath("/scratch", ENV["USER"]) : pwd())

    SCRATCH[] = expanduser(SCRATCH[])
    SCRATCH_kerneldir[] = joinpath(SCRATCH[], "kernels")

    if !isdir(SCRATCH_kerneldir[])
        @info "Creating $(SCRATCH_kerneldir[])"
        mkdir(SCRATCH_kerneldir[])
    end
end

function rank_size(comm)
	rank = MPI.Comm_rank(comm)
	np = MPI.Comm_size(comm)
	rank, np
end
function productsplit(iters, comm)
	rank, np = rank_size(comm)
	ProductSplit(iters, np, rank + 1)
end

include("$(@__DIR__)/pointsonasphere.jl")
include("$(@__DIR__)/finite_difference.jl")
include("$(@__DIR__)/continuous_FFT.jl")
include("$(@__DIR__)/directions.jl")
include("$(@__DIR__)/timer_utils.jl")
include("$(@__DIR__)/greenfn.jl")
include("$(@__DIR__)/crosscov.jl")
include("$(@__DIR__)/kernel.jl")
include("$(@__DIR__)/kernel3D.jl")
include("$(@__DIR__)/measurementsvalidation.jl")

end # module
