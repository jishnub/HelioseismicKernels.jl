#################################################################
# Cross covariances and changes in cross covariances
#################################################################
include("$(@__DIR__)/greenfn.jl")

module Crosscov

using Reexport

@reexport using ..Greenfn_radial
@reexport using ..Timer_utils

@reexport using NumericallyIntegrateArrays

using DSP
using DistributedArrays
@reexport using WignerD
@reexport using LegendrePolynomials
@reexport using SphericalHarmonicModes
@reexport using SphericalHarmonicArrays
@reexport using PointsOnASphere
@reexport using TwoPointFunctions
@reexport using EponymTuples

import ..Greenfn_radial: Gfn_path_from_source_radius, radial_grid_index
import ..Directions: line_of_sight_covariant

import ..HelioseismicKernels: firstshmodes

import ..HelioseismicKernels: Powspec
import ParallelUtilities: finalize_except_wherewhence

import PointsOnASphere: SphericalPoint
cosχn1n2(n1::SphericalPoint, n2::SphericalPoint) = cosχ((n1.θ, n1.ϕ), (n2.θ, n2.ϕ))
∂ϕ₂cosχn1n2(n1::SphericalPoint, n2::SphericalPoint) = ∂ϕ₂cosχ((n1.θ, n1.ϕ), (n2.θ, n2.ϕ))

@reexport using Base.Threads

export Cω
export Cϕω
export Cω_onefreq
export hω
export ht
export Powspec
export Ct
export Cω_∂ϕ₂Cω
export ∂ϕ₂Ct
export ∂ϕ₂Cω
export δCω_uniform_rotation_firstborn_integrated_over_angle
export δCω_uniform_rotation_rotatedwaves_linearapprox
export δCω_uniform_rotation_rotatedwaves
export δCt_uniform_rotation_rotatedwaves
export δCt_uniform_rotation_rotatedwaves_linearapprox
export time_window_indices_by_fitting_bounce_peak
export time_window_bounce_filter
export δCω_isotropicδc_C_minus_C0
export δCt_isotropicδc_C_minus_C0
export δCω_isotropicδc_firstborn_integrated_over_angle

export read_parameters_for_points
export ℓ_and_ν_range

export SCRATCH_kerneldir

export @save_to_fits_and_return
export save_to_fits_and_return
export @append_los_if_necessary

abstract type SeismicMeasurement end
struct TravelTimes <: SeismicMeasurement end
struct Amplitudes <: SeismicMeasurement end
export SeismicMeasurement, TravelTimes, Amplitudes

export @def

export shtype
export VSHtype

export computeY₀₀
export computeY₁₀

export pad_zeros_ν

export perturbationparameter
export flows
export soundspeed

abstract type perturbationparameter end
struct soundspeed <: perturbationparameter end
struct flows <: perturbationparameter end

########################################################################################
# Define some useful macros
########################################################################################

# A macro to define other macros
macro def(name, definition)
	return quote
		macro $(esc(name))()
		  esc($(Expr(:quote, definition)))
		end
	end
end

########################################################################################
# Macro to call appropriate 3D method given a 2D one
########################################################################################

macro two_points_on_the_surface(fn)
	callermodule = __module__
	quote
		function $(esc(fn))(nobs1::Point2D,nobs2::Point2D,
			los::los_direction=los_radial();kwargs...)

			r_obs = get(kwargs,:r_obs,r_obs_default)
			xobs1 = Point3D(r_obs,nobs1)
			xobs2 = Point3D(r_obs,nobs2)
			$callermodule.$fn(xobs1,xobs2,los;kwargs...)
		end
	end
end

########################################################################################
# Macro to paste code for reading in Green functions
########################################################################################

@def read_α_r₁ begin
	read_Gfn_file_at_index!(α_r₁,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,r₁_ind,obsindFITS(los),1,1)
end

@def read_α_r₂ begin
	read_Gfn_file_at_index!(α_r₂,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,r₂_ind,obsindFITS(los),1,1)
end

@def read_α_r₁_α_r₂ begin
	@read_α_r₁
	if r₁_ind != r₂_ind
		@read_α_r₂
	end
end

@def read_α_robs begin
	α_robs = read_Gfn_file_at_index(Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,r_obs_ind,obsindFITS(los),1,1)
end

########################################################################################

function line_of_sight_covariant(xobs1::SphericalPoint,xobs2::SphericalPoint,los::los_direction)
	l1 = line_of_sight_covariant(xobs1,los)
	l2 = line_of_sight_covariant(xobs2,los)
	l1,l2
end

function line_of_sight_covariant(nobs1::SphericalPoint,nobs2_arr::Vector{<:SphericalPoint},los::los_direction)
	l1 = line_of_sight_covariant(nobs1,los)
	l2 = [line_of_sight_covariant(nobs2,los) for nobs2 in nobs2_arr]
	l1,l2
end

########################################################################################
# Get the modes to be used
########################################################################################

function ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)
	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ν_ind_range = get(kwargs,:ν_ind_range,axes(ν_arr,1))
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)
	np = ParallelUtilities.nworkersactive(modes_iter.iterators)
	ℓ_range,ν_ind_range,modes_iter,np
end

########################################################################################

@inline shtype(::los_radial) = OSH()
@inline shtype(::los_earth) = GSH()

@inline VSHtype(x1::SphericalPoint, x2::SphericalPoint) = VSHtype(x1.θ,x2.θ)
@inline VSHtype(::Real, ::Real) = PB()
@inline VSHtype(::Equator, ::Equator) = Hansen()

function VSHtype(x1::SphericalPoint, x2arr::Vector{<:SphericalPoint})
	alleq = all(x->isa(x.θ,Equator), x2arr)
	alleq && return Hansen()
	PB()
end

@inline VSHvectorinds(x1::SphericalPoint, x2::SphericalPoint) = VSHvectorinds(VSHtype(x1,x2))
@inline VSHvectorinds(::PB) = -1:1
@inline VSHvectorinds(::Hansen) = 0:1


WignerD.djmatrix!(dj, l, ::Nothing) = nothing

unpackeuler(::Nothing) = (nothing,nothing,nothing)
unpackeuler(eulerangles::NTuple{3,Real}) = eulerangles

export unpackeuler

########################################################################################
# Save location
########################################################################################	

const SCRATCH_kerneldir = joinpath(SCRATCH,"kernels")
if !isdir(SCRATCH_kerneldir)
	mkdir(SCRATCH_kerneldir)
end

########################################################################################
# Generate filenames and save variables
########################################################################################

# Functions to save variable to a file
macro append_los_if_necessary(var,T)
	var_str = string(var)
	ex = quote
		if isa($T,los_radial)
			$var_str
		elseif isa($T,los_earth)
			$var_str*"_los"
		end
	end
	esc(ex)
end

macro save_to_fits_and_return(val)
	val_str = string(val)
	ex = quote
		filename = $val_str*".fits"
		filepath = joinpath($SCRATCH_kerneldir,filename)
		save_to_fits_and_return(filepath,$val)
	end
	esc(ex)
end

macro save_to_fits_and_return(val,T)
	val_str = string(val)
	ex = quote 
		filename = @append_los_if_necessary($val_str,$T)*".fits"
		filepath = joinpath($SCRATCH_kerneldir,filename)
		save_to_fits_and_return(filepath,$val)
	end
	esc(ex)
end

function save_to_fits_and_return(filepath,val::Array{<:Complex})
	FITS(filepath,"w") do f
		write(f,copy(reinterpret(Float64,val)))
	end
	val
end

function save_to_fits_and_return(filepath,val::Array{<:Real})
	FITS(filepath,"w") do f
		write(f,val)
	end
	val
end

function save_to_fits_and_return(filepath,val::Number)
	FITS(filepath,"w") do f
		write(f,copy(reinterpret(Float64,[val])))
	end
	val
end

function save_to_fits_and_return(filepath,val::OffsetArray)
	save_to_fits_and_return(filepath,parent(val))
	val
end

radial_grid_index(x::Point3D) = radial_grid_index(x.r)

Gfn_path_from_source_radius(x::Point3D;kwargs...) = Gfn_path_from_source_radius(x.r;kwargs...)

#######################################################################################################
# Full Frequency axis
#######################################################################################################

function pad_zeros_ν(arr::Array{<:Any,N},ν_ind_range,
	Nν_Gfn,ν_start_zeros,ν_end_zeros,dim::Integer=1) where {N}

	ax_leading  = CartesianIndices(axes(arr)[1:dim-1])
	ax_trailing = CartesianIndices(axes(arr)[dim+1:end])
	N_leading_zeros_ν = ν_start_zeros + first(ν_ind_range) - 1
	N_trailing_zeros_ν = Nν_Gfn - last(ν_ind_range) + ν_end_zeros

	T = Tuple{Vararg{<:AbstractUnitRange,N}}

	inds_leading = (ax_leading.indices...,1:N_leading_zeros_ν,ax_trailing.indices...) :: T
	inds_trailing = (ax_leading.indices...,1:N_trailing_zeros_ν,ax_trailing.indices...) :: T

	lead_arr = zeros(eltype(arr),inds_leading)
	trail_arr = zeros(eltype(arr),inds_trailing)

	cat(lead_arr,arr,trail_arr,dims=dim)
end

# If an OffsetArray is passed we may infer the frequency range from its axes
function pad_zeros_ν(arr::OffsetArray,Nν_Gfn,
	ν_start_zeros,ν_end_zeros,dim::Integer=1)

	pad_zeros_ν(parent(arr),axes(arr,dim),Nν_Gfn,ν_start_zeros,ν_end_zeros,dim)
end

########################################################################################
# Read parameters for source and observation points
########################################################################################

read_parameters_for_points(;kwargs...) = read_all_parameters(;kwargs...)

function read_parameters_for_points(xobs1::Point3D,xobs2::Point3D;kwargs...)
	p_Gsrc = read_all_parameters(;kwargs...)
	p_Gobs1 = read_all_parameters(;kwargs...,r_src=xobs1.r)
	p_Gobs2 = read_all_parameters(;kwargs...,r_src=xobs2.r)
	return p_Gsrc,p_Gobs1,p_Gobs2
end

function read_parameters_for_points(xobs1::Point3D,xobs2_arr::Vector{<:Point3D};kwargs...)
	p_Gsrc = read_all_parameters(;kwargs...)
	p_Gobs1 = read_all_parameters(;kwargs...,r_src=xobs1.r)
	p_Gobs2 = [read_all_parameters(;kwargs...,r_src=xobs2.r) for xobs2 in xobs2_arr]
	return p_Gsrc,p_Gobs1,p_Gobs2
end

function read_parameters_for_points(::Point2D,::Union{Point2D,Vector{<:Point2D}};kwargs...)
	p_Gsrc = read_all_parameters(;kwargs...)
	r_obs = get(kwargs,:r_obs,r_obs_default)
	p_Gobs = read_all_parameters(;kwargs...,r_src=r_obs)
	return p_Gsrc,p_Gobs,p_Gobs
end

function read_parameters_for_points(::Point2D,::Vector{<:Point2D};kwargs...)
	p_Gsrc = read_all_parameters(;kwargs...)
	r_obs = get(kwargs,:r_obs,r_obs_default)
	p_Gobs = read_all_parameters(;kwargs...,r_src=r_obs)
	return p_Gsrc,p_Gobs,p_Gobs
end

function computeY₀₀(::los_radial,xobs1::SphericalPoint,xobs2::SphericalPoint,
	ℓ_range::AbstractRange{Int})

	Pl(cosχn1n2(xobs1,xobs2),lmax=maximum(ℓ_range))
end

function computeY₀₀(::los_radial,xobs1::SphericalPoint,xobs2_arr::Vector{<:SphericalPoint},
	ℓ_range::AbstractRange{Int})
	
	P = zeros(0:maximum(ℓ_range),length(xobs2_arr))
	
	for (x2ind,xobs2) in enumerate(xobs2_arr)
		Pl!(@view(P[:,x2ind]),cosχn1n2(xobs1,xobs2))
	end

	P = permutedims(P)
	[@view(P[:,i]) for i in axes(P,2)]
end

function computeY₀₀(::los_earth,xobs1::SphericalPoint,xobs2::SphericalPoint,ℓ_range::AbstractRange{Int})

	lmax = maximum(ℓ_range)

	Y_d_arrs = WignerD.allocate_Y₁Y₂(GSH(),lmax)
	GSHT = VSHtype(xobs1,xobs2)
	vinds = VSHvectorinds(GSHT)
	modes = LM(SingleValuedRange(0),SingleValuedRange(0))
	Y12 = OffsetVector([SHArray{ComplexF64}((vinds,vinds,modes)) for ℓ in ℓ_range],ℓ_range)

	for ℓ in ℓ_range
		BiPoSH!(GSH(),GSHT,xobs1,xobs2,Y12[ℓ],modes,ℓ,ℓ,Y_d_arrs...)
	end

	return Y12	
end

function computeY₀₀(::los_earth,xobs1::SphericalPoint,xobs2_arr::Vector{<:SphericalPoint},
	ℓ_range::AbstractRange{Int})

	lmax = maximum(ℓ_range)

	Y_d_arrs = WignerD.allocate_Y₁Y₂(GSH(),lmax)
	GSHT = VSHtype(xobs1,xobs2_arr)
	vinds = VSHvectorinds(GSHT)
	modes = LM(SingleValuedRange(0),SingleValuedRange(0))
	Y12 = OffsetVector([[SHArray{ComplexF64}((vinds,vinds,modes)) for _ in xobs2_arr] for ℓ in ℓ_range],ℓ_range)

	for ℓ in ℓ_range
		Y12ℓ = Y12[ℓ]
		for (xobs2_ind,xobs2) in enumerate(xobs2_arr)
			BiPoSH!(GSH(),GSHT,xobs1,xobs2,Y12ℓ[xobs2_ind],modes,ℓ,ℓ,Y_d_arrs...)
		end
	end

	return Y12	
end

function computeY₁₀(::los_radial,xobs1::SphericalPoint,xobs2::SphericalPoint,
	ℓ_range::AbstractRange{Int},GSHT=nothing)

	dPl(cosχn1n2(xobs1,xobs2),lmax=maximum(ℓ_range)).* (∂ϕ₂cosχn1n2(xobs1,xobs2) ::Float64 )
end

function computeY₁₀(::los_radial,xobs1::SphericalPoint,xobs2_arr::Vector{<:SphericalPoint},
	ℓ_range::AbstractRange{Int},GSHT=nothing)

	dP = zeros(0:maximum(ℓ_range),length(xobs2_arr))
	
	for (x2ind,xobs2) in enumerate(xobs2_arr)
		dPl!(@view(dP[:,x2ind]),cosχn1n2(xobs1,xobs2))
		@views dP[:,x2ind] .*= ∂ϕ₂cosχn1n2(xobs1,xobs2) ::Float64
	end

	dP = permutedims(dP)
	[@view(dP[:,i]) for i in axes(dP,2)]
end

function computeY₁₀(::los_earth,xobs1::SphericalPoint,xobs2::SphericalPoint,
	ℓ_range::AbstractRange{Int},GSHT::WignerD.GSHType=PB())

	lmax = maximum(ℓ_range)

	Y_d_arrs = WignerD.allocate_Y₁Y₂(GSH(),lmax)
	GSHT = VSHtype(xobs1,xobs2)
	vinds = VSHvectorinds(GSHT)
	modes = LM(SingleValuedRange(1),SingleValuedRange(0))
	Y12 = OffsetVector([SHArray{ComplexF64}((vinds,vinds,modes)) for ℓ in ℓ_range],ℓ_range)

	for ℓ in ℓ_range
		BiPoSH!(GSH(),GSHT,xobs1,xobs2,Y12[ℓ],modes,ℓ,ℓ,Y_d_arrs...)
	end

	return Y12
end

function computeY₁₀(::los_earth,xobs1::SphericalPoint,xobs2_arr::Vector{<:SphericalPoint},
	ℓ_range::AbstractRange{Int},GSHT::WignerD.GSHType=PB())

	lmax = maximum(ℓ_range)

	Y_d_arrs = WignerD.allocate_Y₁Y₂(GSH(),lmax)
	GSHT = VSHtype(xobs1,xobs2_arr)
	vinds = VSHvectorinds(GSHT)
	modes = LM(SingleValuedRange(1),SingleValuedRange(0))
	Y12 = OffsetVector([[SHArray{ComplexF64}((vinds,vinds,modes)) for _ in xobs2_arr] for ℓ in ℓ_range],ℓ_range)

	for ℓ in ℓ_range
		Y12ℓ = Y12[ℓ]
		for (xobs2_ind,xobs2) in enumerate(xobs2_arr)
			BiPoSH!(GSH(),GSHT,xobs1,xobs2,Y12ℓ[xobs2_ind],modes,ℓ,ℓ,Y_d_arrs...)
		end
	end

	return Y12	
end

maybeupdatedjmatrix!(::los_radial, dj, ℓ, β) = nothing
maybeupdatedjmatrix!(::los_earth, dj, ℓ, β) = djmatrix!(dj, ℓ, β)

maybeupdateDjmatrix(::los_radial, Dlmn, αrot, dj, γrot) = Dlmn
function maybeupdateDjmatrix(::los_earth, Dlmn, αrot, dj, γrot)
	WignerDMatrix(eltype(Dlmn), αrot, dj, γrot)
end

WignerD.WignerDMatrix(::Type, ℓ::Integer, ::Nothing, dj::WignerdMatrix, ::Nothing) = nothing
WignerD.WignerDMatrix(::Type, ::Nothing, dj::WignerdMatrix, ::Nothing) = nothing
WignerD.WignerDMatrix(ℓ::Integer, ::Nothing, dj::WignerdMatrix, ::Nothing) = nothing
WignerD.WignerdMatrix(::Type, ℓ::Integer, ::Nothing) = nothing
WignerD.WignerdMatrix(ℓ::Integer, ::Nothing) = nothing

########################################################################################

function allocateGfn(los::los_direction,obs_at_same_height::Bool)
	α_r₁ = zeros_Float64_to_ComplexF64(obsindG(los)...)
	α_r₂ = obs_at_same_height ? α_r₁ : zeros_Float64_to_ComplexF64(obsindG(los)...)
	@eponymtuple(α_r₁,α_r₂)
end

########################################################################################
# cross-covariances
########################################################################################

function Cωℓ(::los_radial,ω,ℓ,α_r₁::AbstractArray{ComplexF64,0},
	α_r₂::AbstractArray{ComplexF64,0},Pl,l1 = nothing,l2 = nothing,Dlmn = nothing)

	# Dlmn has no effect as this is a scalar

	ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁[]) * α_r₂[] * Pl
end

function Cωℓ(::los_earth,ω,ℓ,α_r₁::AbstractArray{ComplexF64,1},
	α_r₂::AbstractArray{ComplexF64,1},Y12::SHArray{ComplexF64,3},
	l1,l2,Dlmn) where {T}

	# Dlmn has no effect on the tensor C as it's a scalar (l=0,m=0).
	# However the line-of-sight vectors change
	
	pre = ω^2*Powspec(ω) * (-1)^ℓ * √(2ℓ+1)
	s = zero(ComplexF64)
	ind00 = modeindex(firstshmodes(Y12),(0,0))

	@inbounds for β in axes(Y12,2), α in axes(Y12,1)
		s += pre * conj(α_r₁[abs(α)])*α_r₂[abs(β)]*
				l1[α]*l2[β]*Y12[α,β,ind00]
	end
	s
end

function Cω_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_direction,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,c_scale=1,
	eulerangles::Union{Nothing,NTuple{3,Real}}=nothing,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=c_scale)
	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,(ℓ_arr,1:Nν_Gfn),
						ℓ_ωind_iter_on_proc,NGfn_files_src)

	ν_ind_range = ℓ_ωind_iter_on_proc.iterators[end]

	Cω_proc = zeros(ComplexF64,ν_ind_range)

	ℓ_range = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	Y12 = computeY₀₀(los,xobs1,xobs2,ℓ_range)

	ℓ_max = maximum(ℓ_range)

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	l1,l2 = line_of_sight_covariant.((xobs1,xobs2),(los,),(eulerangles,))

	@unpack α_r₁,α_r₂ = allocateGfn(los,r₁_ind == r₂_ind)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]

		@timeit localtimer "FITS" begin
			@read_α_r₁_α_r₂
		end

		@timeit localtimer "Cω calculation" begin
			Cω_proc[ω_ind] += Cωℓ(los,ω,ℓ,α_r₁,α_r₂,Y12[ℓ],l1,l2,nothing)
		end

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	parent(Cω_proc)
end

function Cω_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	nobs1::Point2D,nobs2_arr::Vector{<:Point2D},los::los_direction,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=r_obs_default,c_scale=1,
	eulerangles::Nothing=nothing,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=c_scale)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ν_ind_range = ℓ_ωind_iter_on_proc.iterators[end]

	r₁_ind = radial_grid_index(r_obs)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
		(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	@unpack α_r₁, = allocateGfn(los,true)

	Cω_proc = zeros(ComplexF64,length(nobs2_arr),ν_ind_range)

	ℓ_range = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	Y12 = computeY₀₀(los,nobs1,nobs2_arr,ℓ_range)

	# covariant components
	l1,l2 = line_of_sight_covariant(nobs1,nobs2_arr,los)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]
		Y12ℓ = Y12[ℓ]

		@read_α_r₁

		for n2ind in axes(Cω_proc,1)
			Cω_proc[n2ind,ω_ind] += Cωℓ(los,ω,ℓ,α_r₁,α_r₁,Y12ℓ[n2ind],l1,l2[n2ind])
		end

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	permutedims(parent(Cω_proc))
end

########################################################################################
# Functions that iterate over the modes in parallel
########################################################################################

function _Cω(args...;kwargs...)
	r_src,r_obs,c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(r_src=r_src,c_scale=c_scale)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	print_timings = get(kwargs,:print_timings,false)
	eulerangles = get(kwargs,:eulerangles,nothing)

	Cω_in_range = pmapsum_timed(Cω_partial,modes_iter,
		args...,p_Gsrc,r_src,r_obs,c_scale,eulerangles;
		progress_str="Modes summed in Cω : ",
		print_timings=print_timings)

	pad_zeros_ν(Cω_in_range,ν_ind_range,Nν_Gfn,ν_start_zeros,ν_end_zeros)
end

# With or without los, 3D points
function Cω(xobs1::Point3D,xobs2::Point3D,los::los_direction=los_radial();kwargs...)
	Cω = _Cω(xobs1,xobs2,los;kwargs...)
	eulerangles = get(kwargs,:eulerangles,nothing)
	tag = isnothing(eulerangles) ? "" : "_rot"
	FITS(joinpath(SCRATCH_kerneldir,"Cω"*tag*".fits"),"w") do f
		write(f,reinterpret(Float64,parent(Cω)))
	end
	Cω
end

# With or without los, 2D points
@two_points_on_the_surface Cω

# With or without los, 2D/3D points, single frequency
function Cω(xobs1,xobs2,ν::Real,los::los_direction=los_radial();kwargs...)
	r_src = get(kwargs,:r_src,r_src_default)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack ν_arr,ν_start_zeros = p_Gsrc

	ν_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_range = ν_ind:ν_ind

	Cω(xobs1,xobs2,los;
		kwargs...,ν_ind_range=ν_ind_range)[ν_ind+ν_start_zeros]
end

# Multiple 2D points
function Cω(nobs1::Point2D,nobs2_arr::Vector{<:Point2D},
	los::los_direction=los_radial();kwargs...)

	Cω_n2arr = _Cω(nobs1,nobs2_arr,los;kwargs...)
	@save_to_fits_and_return(Cω_n2arr)
end

########################################################################################################
# Spectrum of C(ℓ,ω)
########################################################################################################    

function Cωℓ_spectrum_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	r_obs,los::los_radial,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	r_obs_ind = radial_grid_index(r_obs)
	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src)
	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,Nν_Gfn,ω_arr = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,(ℓ_arr,1:Nν_Gfn),
						ℓ_ωind_iter_on_proc,NGfn_files_src)

	@unpack α_r₁, = allocateGfn(los,true)
	r₁_ind = radial_grid_index(r_obs)

	Cℓω = zeros(ℓ_range,ν_ind_range)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		# get the only element of a 0-dim array
		@read_α_r₁

		# m-averaged, so divided by 2ℓ+1
		Cℓω[ℓ,ω_ind] = ω^2 * Powspec(ω) * 1/4π * abs2(α_r₁[])

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	permutedims(parent(Cℓω))
end

function Cωℓ_spectrum(;kwargs...)
	r_src,r_obs,c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(r_src=r_src,c_scale=c_scale)
	@unpack ℓ_arr,ν_arr = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	Cωℓ_in_range = pmapsum_timed(Cωℓ_spectrum_partial,modes_iter,
		r_obs,los_radial(),p_Gsrc,r_src;
		progress_str = "Modes summed in Cωℓ_spectrum : ",
		print_timings = get(kwargs,:print_timings,false))
	
	@save_to_fits_and_return(Cωℓ_in_range)
end

function Cωℓ_spectrum(ν::Real;kwargs...)
	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack ν_arr = p_Gsrc
	ν_test_ind = searchsortedfirst(ν_arr,ν)

	Cωℓ_spectrum(;ν_ind_range=ν_test_ind:ν_test_ind,kwargs...)
end

########################################################################################################
# Derivatives of cross-covariance (useful mainly in the radial case)
########################################################################################################

function ∂ϕ₂Cω_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_radial,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,c_scale=1,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=c_scale)
	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	Cω_proc = zeros(ComplexF64,ν_ind_range)

	∂ϕ₂Pl_cosχ = dPl(cosχn1n2(xobs1,xobs2),
					lmax=maximum(ℓ_arr)) .* ∂ϕ₂cosχn1n2(xobs1,xobs2)

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	@unpack α_r₁,α_r₂ = allocateGfn(los,r₁_ind == r₂_ind)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
		
		ω = ω_arr[ω_ind]
		
		@read_α_r₁_α_r₂

		Cω_proc[ω_ind] += Cωℓ(los_radial(), ω, ℓ, α_r₁, α_r₂, ∂ϕ₂Pl_cosχ[ℓ])

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	parent(Cω_proc)
end

function ∂ϕ₂Cω_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2_arr::Vector{<:Point3D},los::los_radial,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,c_scale=1,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=c_scale)
	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ν_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	Cω_proc = zeros(ComplexF64,ν_ind_range)

	∂ϕ₂Pl_cosχ = zeros(0:maximum(ℓ_arr),length(xobs2_arr))
	@inbounds for (x2ind,xobs2) in enumerate(xobs2_arr)
		dPl_x2 = view(∂ϕ₂Pl_cosχ,:,x2ind)
		dPl!(dPl_x2,cosχn1n2(xobs1,xobs2))
		dPl_x2 .*= ∂ϕ₂cosχn1n2(xobs1,xobs2)
	end

	∂ϕ₂Pl_cosχ = permutedims(∂ϕ₂Pl_cosχ)

	r₁_ind = radial_grid_index(xobs1)
	r₂_ind_prev = radial_grid_index(first(xobs2_arr))

	@unpack α_r₁,α_r₂ = allocateGfn(los,false)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		@read_α_r₁

		for (x2ind,xobs2) in enumerate(xobs2_arr)
			r₂_ind = radial_grid_index(xobs2)

			if (x2ind==1 || r₂_ind != r₂_ind_prev) && r₂_ind != r₁_ind
				@read_α_r₂
			end

			Cω_proc[x2ind,ω_ind] += Cωℓ(los, ω, ℓ, α_r₁, α_r₂, ∂ϕ₂Pl_cosχ[x2ind,ℓ])

			r₂_ind_prev = r₂_ind
		end

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	permutedims(parent(Cω_proc))
end

# Without los, 3D points
function ∂ϕ₂Cω(xobs1::Point3D,xobs2::Union{Point3D,Vector{<:Point3D}},
	los::los_radial=los_radial();kwargs...)
	
	r_src,r_obs,c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(r_src=r_src,c_scale=c_scale)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	print_timings = get(kwargs,:print_timings,false)

	Cω_in_range = pmapsum_timed(∂ϕ₂Cω_partial,modes_iter,
			xobs1,xobs2,los,p_Gsrc,r_src,r_obs,c_scale;
			progress_str="Modes summed in ∂ϕ₂Cω : ",
			print_timings = print_timings)

	∂ϕ₂Cω = pad_zeros_ν(Cω_in_range,
		ν_ind_range,Nν_Gfn,ν_start_zeros,ν_end_zeros)

	@save_to_fits_and_return(∂ϕ₂Cω)
end

# With or without los, 2D points
@two_points_on_the_surface ∂ϕ₂Cω

# With or without los, multiple 2D points
function ∂ϕ₂Cω(nobs1::Point2D,nobs2_arr::Vector{<:Point2D},
	args...;kwargs...)

	r_obs = get(kwargs,:r_obs,r_obs_default)
	xobs1 = Point3D(r_obs,nobs1)
	xobs2_arr = [Point3D(r_obs,nobs2) for nobs2 in nobs2_arr]
	∂ϕ₂Cω(xobs1,xobs2_arr,args...;kwargs...)
end

function Cω_∂ϕ₂Cω_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_radial,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,c_scale=1,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=c_scale)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	Cω_proc = zeros(ComplexF64,0:1,ν_ind_range)

	Pl_cosχ,∂ϕ₂Pl_cosχ = Pl_dPl(cosχn1n2(xobs1,xobs2),lmax=maximum(ℓ_arr))
	∂ϕ₂Pl_cosχ .*= ∂ϕ₂cosχn1n2(xobs1,xobs2)

	@unpack α_r₁,α_r₂ = allocateGfn(los,r₁_ind == r₂_ind)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
		
		ω = ω_arr[ω_ind]

		@read_α_r₁_α_r₂
		
		f = Cωℓ(los_radial(), ω, ℓ, α_r₁, α_r₂, 1)

		Cω_proc[0,ω_ind] += f * Pl_cosχ[ℓ]
		Cω_proc[1,ω_ind] += f * ∂ϕ₂Pl_cosχ[ℓ]

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	permutedims(parent(Cω_proc))
end

function Cω_∂ϕ₂Cω_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	nobs1::Point2D,nobs2_arr::Vector{<:Point2D},los::los_radial,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=r_obs_default,c_scale=1,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=c_scale)
	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	Cω_proc = zeros(ComplexF64,0:1,length(nobs2_arr),ν_ind_range)

	r_obs_ind = radial_grid_index(r_obs)

	lmax = maximum(ℓ_arr)

	∂ϕ₂Pl_cosχ = zeros(0:lmax,length(nobs2_arr))
	Pl_cosχ = zeros(0:lmax,length(nobs2_arr))
	
	P = zeros(0:lmax,0:1)

	for (n2ind,nobs2) in enumerate(nobs2_arr)
		Pl_dPl!(P,cosχn1n2(nobs1,nobs2))
		@. Pl_cosχ[:,n2ind] = @view(P[:,0])
		∂ϕ₂Pl_cosχ[:,n2ind] .= @view(P[:,1]) .* ∂ϕ₂cosχn1n2(nobs1,nobs2)
	end

	Pl_cosχ = permutedims(Pl_cosχ)
	∂ϕ₂Pl_cosχ = permutedims(∂ϕ₂Pl_cosχ)

	@unpack α_r₁,α_r₂ = allocateGfn(los,true)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
		
		ω = ω_arr[ω_ind]
		
		@read_α_r₁

		f = Cωℓ(los_radial(),ω,ℓ,α_r₁,α_r₁,1,nothing,nothing) # we'll multiply the angular part later

		for n2ind in axes(Cω_proc,2)
			Cω_proc[0,n2ind,ω_ind] += f * Pl_cosχ[n2ind,ℓ]
			Cω_proc[1,n2ind,ω_ind] += f * ∂ϕ₂Pl_cosχ[n2ind,ℓ]
		end

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	permutedims(parent(Cω_proc),[3,2,1])
end

function _Cω_∂ϕ₂Cω(args...;kwargs...)
	r_src,r_obs,c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	Cω_in_range = pmapsum_timed(Cω_∂ϕ₂Cω_partial,modes_iter,
			args...,p_Gsrc,r_src,r_obs,c_scale;
			progress_str="Modes summed in Cω_∂ϕ₂Cω : ",kwargs...)

	Cω_∂ϕ₂Cω = pad_zeros_ν(Cω_in_range,
			ν_ind_range,Nν_Gfn,ν_start_zeros,ν_end_zeros)
end

# Without los, 3D points
function Cω_∂ϕ₂Cω(xobs1::Point3D,xobs2::Point3D,
	los::los_radial=los_radial();kwargs...)
	
	Cω_∂ϕ₂Cω = _Cω_∂ϕ₂Cω(xobs1,xobs2,los;kwargs...)
	return Cω_∂ϕ₂Cω[:,1],Cω_∂ϕ₂Cω[:,2]
end

# With or without los, 2D points
@two_points_on_the_surface Cω_∂ϕ₂Cω

# Without los, multiple 2D points
function Cω_∂ϕ₂Cω(nobs1::Point2D,nobs2_arr::Vector{<:Point2D},
	los::los_radial=los_radial();kwargs...)

	Cω_∂ϕ₂Cω = _Cω_∂ϕ₂Cω(nobs1,nobs2_arr,los;kwargs...)
	return Cω_∂ϕ₂Cω[:,:,1],Cω_∂ϕ₂Cω[:,:,2]
end

########################################################################################################
# Time-domain cross-covariance
########################################################################################################

function Ct(xobs1::SphericalPoint,xobs2::SphericalPoint,
	los::los_direction=los_radial();kwargs...)

	C = Cω(xobs1,xobs2,los;kwargs...)
	Ct(C,los;kwargs...)
end

function Ct(Δϕ::Real,los::los_direction=los_radial();kwargs...)
	nobs1 = Point2D(π/2,0)
	nobs2 = Point2D(π/2,Δϕ)
	Ct(nobs1,nobs2,los;kwargs...)
end

function Ct(Cω_arr::AbstractArray{ComplexF64},los::los_direction=los_radial();kwargs...)
	dν = get(kwargs,:dν) do
		read_parameters("dν";kwargs...)[1]
	end
	
	τ_ind_range = get(kwargs,:τ_ind_range,Colon())

	Ct = fft_ω_to_t(Cω_arr,dν)

	if !(τ_ind_range isa Colon)
		Ct = Ct[τ_ind_range,..]
	end

	eulerangles = get(kwargs,:eulerangles,nothing)
	tag = isnothing(eulerangles) ? "" : "_rot"
	FITS(joinpath(SCRATCH_kerneldir,"Ct"*tag*".fits"),"w") do f
		write(f,Ct)
	end
	Ct
end

function ∂ϕ₂Ct(xobs1::SphericalPoint,xobs2::SphericalPoint,
	los::los_direction=los_radial();kwargs...)

	C = ∂ϕ₂Cω(xobs1,xobs2,los;kwargs...)
	∂ϕ₂Ct(C,los;kwargs...)
end

function ∂ϕ₂Ct(∂ϕ₂Cω_arr::AbstractArray{ComplexF64},
	los::los_direction=los_radial();kwargs...) 

	dν = get(kwargs,:dν) do
		read_parameters("dν";kwargs...)[1]
	end

	τ_ind_range = get(kwargs,:τ_ind_range,Colon())

	∂ϕCt = fft_ω_to_t(∂ϕ₂Cω_arr,dν)
	
	if !(τ_ind_range isa Colon)
		∂ϕCt = ∂ϕCt[τ_ind_range,..]
	end
	@save_to_fits_and_return(∂ϕCt,los)
end

########################################################################################################
# Cross-covariance at all distances on the equator, essentially the time-distance diagram
########################################################################################################

function CΔϕω_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	r₁,r₂,::los_radial,Δϕ_arr,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,c_scale=1,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing;kwargs...)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=c_scale)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	r₁_ind,r₂_ind = radial_grid_index.((r₁,r₂))

	nϕ = length(Δϕ_arr)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,(ℓ_arr,1:Nν_Gfn),
						ℓ_ωind_iter_on_proc,NGfn_files_src)

	Cϕω_arr = zeros(ComplexF64,nϕ,ν_ind_range)

	Pl_cosχ = zeros(0:maximum(ℓ_arr),nϕ)

	for (ϕ_ind,Δϕ) in enumerate(Δϕ_arr)
		Pl!(view(Pl_cosχ,:,ϕ_ind),cos(Δϕ))
	end

	Pl_cosχ = permutedims(Pl_cosχ)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
		
		ω = ω_arr[ω_ind]

		@read_α_r₁_α_r₂

		for ϕ_ind in 1:nϕ
			Cϕω_arr[ϕ_ind,ω_ind] += Cωℓ(los_radial(),ω,ℓ,α_r₁,α_r₂,Pl_cosχ[ϕ_ind,ℓ])
		end

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	parent(Cϕω_arr)
end

function CΔϕω_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,r₂,::los_earth,Δϕ_arr,p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,c_scale=1,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing;kwargs...)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=c_scale)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ν_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,r₂))

	nϕ = length(Δϕ_arr)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,(ℓ_arr,1:Nν_Gfn),
						ℓ_ωind_iter_on_proc,NGfn_files_src)

	Cϕω_arr = zeros(ComplexF64,nϕ,ν_ind_range)

	# covariant components
	l1 = conj.(line_of_sight(xobs1))

	ℓ_ωind_iter_on_proc = sort(collect(ℓ_ωind_iter_on_proc))

	lmax = maximum(ℓ_arr)
	Yobs1 = zeros(ComplexF64,-lmax:lmax,-1:1)
	dℓθ = zeros(axes(Yobs1))

	Yobs2 = zeros(ComplexF64,-lmax:lmax,-1:1)

	l1Y1starl2Y2 = zeros(ComplexF64,axes(Yobs1,2),axes(Yobs2,2))

	ℓ_prev = 0

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		if ℓ != ℓ_prev
			Ylmatrix!(Yobs1,dℓθ,ℓ,xobs1,compute_d_matrix=true)
		end
		
		ω = ω_arr[ω_ind]

		α_r₁ = read_Gfn_file_at_index(Gfn_fits_files_src,
			ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),NGfn_files_src,r₁_ind,1:2,1,1)
		α_r₂ = α_r₁
		if r₁_ind != r₂_ind
			α_r₂ = read_Gfn_file_at_index(Gfn_fits_files_src,
			ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),NGfn_files_src,r₂_ind,1:2,1,1)
		end

		@inbounds for (ϕ_ind,ϕ) in enumerate(Δϕ_arr)
			xobs2 = Point3D(r₂,xobs1.θ,ϕ)
			Ylmatrix!(Yobs2,dℓθ,ℓ,(xobs1.θ,ϕ))
			fill!(l1Y1starl2Y2,zero(ComplexF64))
			l2 = conj.(line_of_sight(xobs2))
			@inbounds for β in -1:1,α in -1:1
				@inbounds for m in axes(Yobs1,1)
					l1Y1starl2Y2[α,β] += l1[α]*l2[β]*conj(Yobs1[m,α])*Yobs2[m,β]
				end
				Cϕω_arr[ϕ_ind,ω_ind] += ω^2 * Powspec(ω) *
										conj(α_r₁[abs(α)]) * 
										α_r₂[abs(β)] * 
										l1Y1starl2Y2[α,β]
			end
		end

		ℓ_prev = ℓ

		signaltomaster!(progress_channel)

	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	parent(Cϕω_arr)
end

function _CΔϕω(args...;kwargs...)
	r_src,r_obs,c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)
	lmax = maximum(ℓ_range)

	Δϕ_arr = get(kwargs,:Δϕ_arr,LinRange(0,π,lmax+1)[1:end-1])
	
	Cω_in_range = pmapsum_timed(CΔϕω_partial,modes_iter,
			args...,Δϕ_arr,p_Gsrc,r_src,c_scale;
			progress_str="Modes summed in CΔϕω : ",kwargs...)

	CΔϕω = pad_zeros_ν(Cω_in_range,
				ν_ind_range,Nν_Gfn,ν_start_zeros,ν_end_zeros,2)
end

function CΔϕω(r₁::Real=r_obs_default,r₂::Real=r_obs_default,
	los::los_radial=los_radial();kwargs...)

	CΔϕω = _CΔϕω(r₁,r₂,los;kwargs...)
	@save_to_fits_and_return(CΔϕω)
end

CΔϕω(xobs1::Point3D,args...;kwargs...) = CΔϕω(xobs1.r,args...;kwargs...)
CΔϕω(xobs1::Point3D,xobs2::Point3D,args...;kwargs...) = 
	CΔϕω(xobs1.r,xobs2.r,args...;kwargs...)

function CΔϕω(xobs1::Point3D,r₂::Real,::los_earth;kwargs...)
	CΔϕω_los = _CΔϕω(xobs1,r₂,los_earth();kwargs...)
	@save_to_fits_and_return(CΔϕω_los)
end

function CtΔϕ(xobs1=r_obs_default,xobs2=r_obs_default,
	los::los_direction=los_radial();kwargs...)
	
	dν = get(kwargs,:dν) do
		read_parameters("dν";kwargs...)[1]
	end

	CωΔϕ = permutedims(CΔϕω(xobs1,xobs2,los;kwargs...))

	CtΔϕ = fft_ω_to_t(CωΔϕ,dν)

	τ_ind_range = get(kwargs,:τ_ind_range,Colon())
	if !(τ_ind_range isa Colon)
		CtΔϕ = CtΔϕ[τ_ind_range,..]
	end

	@save_to_fits_and_return(CtΔϕ,los)
end

CtΔϕ(xobs1::Point3D,los::los_direction=los_radial();kwargs...) = 
	CtΔϕ(xobs1,r_obs_default,los;kwargs...)

function CΔϕt(xobs1,xobs2,los::los_direction=los_radial();kwargs...)
	CΔϕt = permutedims(CtΔϕ(xobs1,xobs2,los;kwargs...))
	@save_to_fits_and_return(CtΔϕ,los)
end

CΔϕt(xobs1,los::los_direction=los_radial();kwargs...) = 
	CΔϕt(xobs1,r_obs_default,los;kwargs...)

########################################################################################################
# Cross-covariance in a rotating frame
########################################################################################################

function Cτ_rotating_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,::los_radial,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,c_scale=1,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=c_scale)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,(ℓ_arr,1:Nν_Gfn),
						ℓ_ωind_iter_on_proc,NGfn_files_src)

	α₁conjα₂ωℓ = zeros(ComplexF64,ℓ_range,ν_ind_range)

	α_r₁ = zeros_Float64_to_ComplexF64()
	α_r₂ = (r₁_ind != r₂_ind) ? zeros_Float64_to_ComplexF64() : α_r₁

	# Read all radial parts
	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		read_Gfn_file_at_index!(α_r₁,Gfn_fits_files_src,
				(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,r₁_ind,1,1,1)

		conjα₁_α₂ = Complex(abs2(α_r₁[]))

		if r₁_ind != r₂_ind
			read_Gfn_file_at_index!(α_r₂,Gfn_fits_files_src,
				(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,r₂_ind,1,1,1)

			conjα₁_α₂ = α_r₁[]*conj(α_r₂[])
		end

		α₁conjα₂ωℓ[ℓ,ω_ind] = conjα₁_α₂ * ω^2*Powspec(ω)

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	permutedims(parent(α₁conjα₂ωℓ))
end

# Without los, radial components, 3D points
function Cτ_rotating(xobs1::Point3D,xobs2::Point3D,los::los_radial=los_radial();kwargs...)
	
	# Return C(Δϕ,ω) = RFFT(C(Δϕ,τ)) = RFFT(IRFFT(C0(Δϕ - Ωτ,ω))(τ))
	r_src,r_obs,c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros,Nt,dt,dν = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)
	τ_ind_range = get(kwargs,:τ_ind_range,1:Nt)		

	ℓ_range,ν_ind_range,modes_iter,np = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)
	lmax = maximum(ℓ_range)

	Nτ = length(τ_ind_range)

	# The first step is loading in the αωℓ
	# Fourier transform the αωℓ and transpose to obtain αℓt
	Cω_in_range = pmapsum_timed(Cτ_rotating_partial,modes_iter,
			xobs1,xobs2,los,p_Gsrc,r_src,c_scale;
			progress_str="Modes summed in Cτ : ")

	αωℓ = pad_zeros_ν(Cω_in_range,ν_ind_range,Nν_Gfn,ν_start_zeros,ν_end_zeros)

	αtℓ = fft_ω_to_t(αωℓ,dν)
	αℓt = permutedims(αtℓ)[:,τ_ind_range]

	np = min(nworkers(),Nτ)

	αℓt = distribute(αℓt,procs=workers()[1:np],dist=[1,np])

	τ_ind_tracker = RemoteChannel(()->Channel{Bool}(length(τ_ind_range)))
	progress_bar_τ = Progress(length(τ_ind_range),1,"Cτ : ")

	Cτ_arr = nothing

	@sync begin
		@async begin
			Cτ_arr = DArray((Nτ,),workers()[1:np],[np]) do inds
				τ_ind_range_proc = τ_ind_range[first(inds)]
				αℓt_local = OffsetArray(αℓt[:lp],ℓ_range,τ_ind_range_proc)
				Cτ_arr = zeros(τ_ind_range_proc)
				for τ_ind in τ_ind_range_proc
					# τ goes from -T/2 to T/2-dt but is fftshifted
					τ = (τ_ind<=div(Nt,2) ? (τ_ind-1) : (τ_ind-1- Nt)) * dt
					xobs2′ = Point3D(xobs2.r,xobs2.θ,xobs2.ϕ-Ω_rot*τ)
					Pl_cosχ = Pl(cosχn1n2(xobs1,xobs2′),lmax=lmax)
					for ℓ in ℓ_range
						Cτ_arr[τ_ind] += (2ℓ+1)/4π * αℓt_local[ℓ,τ_ind] * Pl_cosχ[ℓ]
					end
					put!(τ_ind_tracker,true)
				end
				finalize(τ_ind_tracker)
				parent(Cτ_arr)
			end
			put!(τ_ind_tracker,false)
			finish!(progress_bar_τ)
		end
		@async begin
			while take!(τ_ind_tracker)
				next!(progress_bar_τ)
			end
		end
	end

	OffsetArray(Array(Cτ_arr),τ_ind_range)
end

# 2D points
@two_points_on_the_surface Cτ_rotating

# Without los, radial components, multiple 3D points
function Cτ_rotating(xobs1,xobs2_arr::Vector{T},los::los_direction=los_radial();
	Ω_rot = 20e2/Rsun,τ_ind_range = nothing,kwargs...) where {T<:SphericalPoint}
	
	# Return C(Δϕ,ω) = RFFT(C(Δϕ,τ)) = RFFT(IRFFT(C0(Δϕ - Ωτ,ω))(τ))
	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack Nt,dt = p_Gsrc

	if isnothing(τ_ind_range)
		τ_ind_range = [1:div(Nt,2) for xobs2 in xobs2_arr]
	end

	τ_ind_max_span = minimum(minimum.(τ_ind_range)):maximum(maximum.(τ_ind_range))

	Cτ_arr = [zeros(τ_inds) for τ_inds in τ_ind_range]
	
	@showprogress 1 "Cτ τ ind : " for τ_ind in τ_ind_max_span
		τ = (τ_ind-1) * dt
		if T<:Point3D
			xobs2′_arr = [T(xobs2.r,xobs2.θ,xobs2.ϕ-Ω_rot*τ) for (rank,xobs2) in enumerate(xobs2_arr)
						 if τ_ind in τ_ind_range[rank]]
		elseif T<:Point2D
			xobs2′_arr = [T(xobs2.θ,xobs2.ϕ-Ω_rot*τ) for (rank,xobs2) in enumerate(xobs2_arr)
						 if τ_ind in τ_ind_range[rank]]
		end

		xobs2′inds_arr = [rank for (rank,xobs2) in enumerate(xobs2_arr) if τ_ind in τ_ind_range[rank]]
		Ct_xobs2_arr = Ct(xobs1,xobs2′_arr,los;kwargs...)[τ_ind,:]
		
		for (Ct_x2,x2ind) in zip(Ct_xobs2_arr,xobs2′inds_arr)
			Cτ_arr[x2ind][τ_ind] = Ct_x2
		end
	end

	return Cτ_arr
end

#######################################################################################################################################
# First Born approximation for flows
#######################################################################################################################################

function δCω_uniform_rotation_firstborn_integrated_over_angle_partial(
	ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_radial,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,Ω_rot=20e2/Rsun,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src)
	p_Gobs1 = read_all_parameters(p_Gobs1,r_src=xobs1.r)
	p_Gobs2 = read_all_parameters(p_Gobs2,r_src=xobs2.r)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	Gfn_path_obs1,NGfn_files_obs1 =  p_Gobs1.path,p_Gobs1.num_procs
	Gfn_path_obs2,NGfn_files_obs2 = p_Gobs2.path,p_Gobs2.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2 = 
			Gfn_fits_files((Gfn_path_src,Gfn_path_obs1,Gfn_path_obs2),
			(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc::ProductSplit,
			(NGfn_files_src,NGfn_files_obs1,NGfn_files_obs2))

	δC_r = zeros(ComplexF64,nr,ν_ind_range)

	Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1)
	Gobs1 = zeros_Float64_to_ComplexF64(1:nr,0:1)
	Gobs2 = Gobs1

	f_r₁_rsrc = zeros(ComplexF64,nr)
	f_r₂_rsrc = (r₁_ind != r₂_ind) ? zeros(ComplexF64,nr) : f_r₁_rsrc

	∂ϕ₂Pl_cosχ = dPl(cosχn1n2(xobs1,xobs2),lmax=maximum(ℓ_arr)) .*∂ϕ₂cosχn1n2(xobs1,xobs2)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
		
		ω = ω_arr[ω_ind]
		
		# Green function about source location
			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:2,1,1)

		G_r₁_rsrc = Gsrc[r₁_ind,0]
		G_r₂_rsrc = Gsrc[r₂_ind,0]

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs1,:,1:2,1,1)

		radial_fn_uniform_rotation_firstborn!(f_r₁_rsrc,Gsrc,Gobs1,ℓ,los_radial())

		if r₁_ind != r₂_ind

			# Green function about receiver location
	    	read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs2,:,1:2,1,1)

			radial_fn_uniform_rotation_firstborn!(f_r₂_rsrc,Gsrc,Gobs2,ℓ,los_radial())
		end

		@. δC_r[:,ω_ind] += ω^3*Powspec(ω)* (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * 
					(conj(G_r₁_rsrc) * f_r₂_rsrc + conj(f_r₁_rsrc) * G_r₂_rsrc) 

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)
	closeGfnfits(Gfn_fits_files_obs1)
	closeGfnfits(Gfn_fits_files_obs2)

	δCω = -2im*Ω_rot*simps((@. r^2 * ρ * δC_r),r)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	parent(δCω)
end

function δCω_uniform_rotation_firstborn_integrated_over_angle_partial(
	ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_earth,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,Ω_rot=20e2/Rsun,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src)
	p_Gobs1 = read_all_parameters(p_Gobs1,r_src=xobs1.r)
	p_Gobs2 = read_all_parameters(p_Gobs2,r_src=xobs2.r)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	Gfn_path_obs1,NGfn_files_obs1 =  p_Gobs1.path,p_Gobs1.num_procs
	Gfn_path_obs2,NGfn_files_obs2 = p_Gobs2.path,p_Gobs2.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2 = 
			Gfn_fits_files((Gfn_path_src,Gfn_path_obs1,Gfn_path_obs2),
			(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc::ProductSplit,
			(NGfn_files_src,NGfn_files_obs1,NGfn_files_obs2))

	δC_r = zeros(ComplexF64,nr,ν_ind_range)

	l1,l2 = line_of_sight_covariant(xobs1,xobs2,los)

	Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1,0:1)
	Gobs1 = zeros_Float64_to_ComplexF64(1:nr,0:1,0:1)
	Gobs2 = Gobs1

	f_r₁_rsrc = zeros(ComplexF64,nr,0:1)
	f_r₂_rsrc = (r₁_ind != r₂_ind) ? zeros(ComplexF64,nr,0:1) : f_r₁_rsrc
	
	ℓ_range = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	Y12 = computeY₁₀(los,xobs1,xobs2,ℓ_range)
	vinds = VSHvectorinds(xobs1,xobs2)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]
		
		# Green function about source location
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
		(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:2,1:2,1)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs1,:,1:2,1:2,1)

		radial_fn_uniform_rotation_firstborn!(f_r₁_rsrc,Gsrc,Gobs1,ℓ,los)

		if r₁_ind != r₂_ind

			# Green function about receiver location
	    	read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs2,:,1:2,1:2,1)

			radial_fn_uniform_rotation_firstborn!(f_r₂_rsrc,Gsrc,Gobs2,ℓ,los)
		end

		Y12ℓ = Y12[ℓ]

		@inbounds for β in vinds,γ in vinds
			
			Gγ_r₁_rsrc = Gsrc[r₁_ind,abs(γ),0]
			Gβ_r₂_rsrc = Gsrc[r₂_ind,abs(β),0]

			gγ_r₁_rsrc = view(f_r₁_rsrc,:,abs(γ))
			gβ_r₂_rsrc = view(f_r₂_rsrc,:,abs(β))

			@. δC_r[:,ω_ind] += ω^3*Powspec(ω)*
					(-1)^ℓ * Ω(ℓ,0) * √((2ℓ+1)/6) * 
					(conj(Gγ_r₁_rsrc) * gβ_r₂_rsrc + 
						conj(gγ_r₁_rsrc) * Gβ_r₂_rsrc) * 
						l1[γ]*l2[β]*Y12ℓ[γ,β]
		end

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)
	closeGfnfits(Gfn_fits_files_obs1)
	closeGfnfits(Gfn_fits_files_obs2)

	δCω = -4*Ω_rot*simps((@. r^2 * ρ * δC_r),r)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	parent(δCω)
end

function _δCω_uniform_rotation_firstborn(xobs1,xobs2,args...;kwargs...)
	r_src,r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(r_src=r_src)
	p_Gobs1 = read_all_parameters(r_src=xobs1.r)
	p_Gobs2 = read_all_parameters(r_src=xobs2.r)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	Ω_rot=get(kwargs,:Ω_rot,20e2/Rsun)

	Cω_in_range = pmapsum_timed(
		δCω_uniform_rotation_firstborn_integrated_over_angle_partial,
		modes_iter,xobs1,xobs2,args...,p_Gsrc,p_Gobs1,p_Gobs2,r_src,r_obs,Ω_rot;
		progress_str="Modes summed in δCω_FB : ")

	δCω_flows_FB = pad_zeros_ν(Cω_in_range,
						ν_ind_range,Nν_Gfn,ν_start_zeros,ν_end_zeros)
end

# 3D points
function δCω_uniform_rotation_firstborn_integrated_over_angle(
	xobs1::Point3D,xobs2::Point3D,los::los_direction=los_radial();kwargs...)
	
	δCω_flows_FB = _δCω_uniform_rotation_firstborn(xobs1,xobs2,los;kwargs...)
	@save_to_fits_and_return(δCω_flows_FB)
end

# 2D points
@two_points_on_the_surface δCω_uniform_rotation_firstborn_integrated_over_angle

# With or without los, 3D points, single frequency
function δCω_uniform_rotation_firstborn_integrated_over_angle(
	xobs1,xobs2,ν::Real,los::los_direction=los_radial();kwargs...)
	
	ν_arr,ν_start_zeros = read_parameters("ν_arr","ν_start_zeros";kwargs...)
	ν_test_ind = searchsortedfirst(ν_arr,ν)

	δCω_uniform_rotation_firstborn_integrated_over_angle(
		xobs1,xobs2,los;ν_ind_range=ν_test_ind:ν_test_ind,
		kwargs...)[ν_test_ind + ν_start_zeros]
end

#######################################################################################################################################
# δC(ω) = C(ω) - C0(ω) for flows
########################################################################################################################################

# Linearized, without los, radial components, 3D points
function δCω_uniform_rotation_rotatedwaves_linearapprox(xobs1,xobs2,
	los::los_radial=los_radial();kwargs...)

	# We compute δC(xobs1,xobs2) = -iΩ ∑ℓ (2ℓ+1)/4π ∂ω (ω^2 P(ω) αℓω*(r₂)αℓω(r₁))∂Δϕ Pl(cos(Δϕ))
	
	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src,c_scale=get(kwargs,:c_scale,1))
	@load joinpath(Gfn_path_src,"parameters.jld2") dω

	∂ϕC = ∂ϕ₂Cω(xobs1,xobs2,los;kwargs...)
	Ω_rot= get(kwargs,:Ω_rot,20e2/Rsun)

	∂ω∂ϕC = D(size(∂ϕC,1))*∂ϕC ./ dω

	δCω_flows_rotated_linear = @. -im*Ω_rot*∂ω∂ϕC
	@save_to_fits_and_return(δCω_flows_rotated_linear)
end

# Linearized, without los, radial components, 3D points, single frequency
function δCω_uniform_rotation_rotatedwaves_linearapprox(xobs1,xobs2,ν::Real,
	los::los_radial=los_radial();kwargs...)
	
	# We compute δC(xobs1,xobs2) = -iΩ ∑ℓ (2ℓ+1)/4π ∂ω (ω^2 P(ω) αℓω*(r₂)αℓω(r₁))∂Δϕ Pl(cos(Δϕ))
	
	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src,c_scale=get(kwargs,:c_scale,1))

	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr dω ν_start_zeros

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)

	ν_test_ind = searchsortedfirst(ν_arr,ν)
	ν_on_grid = ν_arr[ν_test_ind]

	ν_ind_range = max(ν_test_ind-7,1):(ν_test_ind+min(7,ν_test_ind-1))
	ν_match_index = searchsortedfirst(ν_ind_range,ν_test_ind)

	∂ϕC = ∂ϕ₂Cω(xobs1,xobs2,los;ν_ind_range=ν_ind_range,
		kwargs...)[ν_ind_range .+ ν_start_zeros]

	∂ω∂ϕC = D(length(∂ϕC))*∂ϕC ./ dω

	δCω_flows_rotated_linear = -im*Ω_rot*∂ω∂ϕC[ν_match_index]
	@save_to_fits_and_return(δCω_flows_rotated_linear)
end

# With or without los, 3D points
function δCω_uniform_rotation_rotatedwaves(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	dt = get(kwargs,:dt) do 
		read_parameters("dt";kwargs...)[1]
	end

	δCt = δCt_uniform_rotation_rotatedwaves(xobs1,xobs2,los;kwargs...)
	δCω_flows_rotated = fft_t_to_ω(δCt,dt)
	@save_to_fits_and_return(δCω_flows_rotated)
end

# With or without los, 3D points, single frequency
function δCω_uniform_rotation_rotatedwaves(xobs1,xobs2,ν::Real,
	los::los_direction=los_radial();kwargs...)

	ν_arr,ν_start_zeros = read_parameters("ν_arr","ν_start_zeros";kwargs...)
	ν_ind = searchsortedfirst(ν_arr,ν)

	δCω_uniform_rotation_rotatedwaves(xobs1,xobs2,los;kwargs...,
		ν_ind_range=ν_ind:ν_ind)[ν_start_zeros + ν_ind]
end

# With or without los, using precomputed time-domain cross-covariance
function δCω_uniform_rotation_rotatedwaves(δCt::AbstractArray;kwargs...)

	dt = get(kwargs,:dt) do 
		read_parameters("dt";kwargs...)[1]
	end
	
	δCω_flows_rotated = fft_t_to_ω(δCt,dt)
	@save_to_fits_and_return(δCω_flows_rotated)
end

#######################################################################################################################################
# δC(t) = C(t) - C0(t) for flows
########################################################################################################################################

# With or without los
function δCt_uniform_rotation_rotatedwaves(xobs1,xobs2,los::los_direction=los_radial();
	kwargs...)

	dν = get(kwargs,:dν) do
		read_parameters("dν";kwargs...)[1]
	end

	C′_t = Cτ_rotating(xobs1,xobs2,los;kwargs...)
	C0_t = get(kwargs,:Ct) do 
		C0_ω = get(kwargs,:Cω) do 
			Cω(xobs1,xobs2,los;kwargs...)
		end
		Ct(C0_ω;dν=dν,kwargs...)
	end
	δCt_flows_rotated = parent(C′_t) .- parent(C0_t)
	@save_to_fits_and_return(δCt_flows_rotated)
end

# With or without los, multiple points
function δCt_uniform_rotation_rotatedwaves(xobs1,xobs2_arr::Vector,
	los::los_direction=los_radial();kwargs...)
	
	C′_t = Cτ_rotating(xobs1,xobs2_arr,los;kwargs...)
	τ_ind_range = [axes(Cx2,1) for Cx2 in C′_t]
	C0_t = get(kwargs,:Ct) do 
			[Ct(xobs1,xobs2_arr,los;kwargs...)[τ_ind,ind] 
			for (ind,τ_ind) in enumerate(τ_ind_range)]
	end
	parent(C′_t) .- C0_t
end

# Linearized, with or without los
function δCt_uniform_rotation_rotatedwaves_linearapprox(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	dν,Nt = read_parameters("dν","Nt";kwargs...)

	# t = (0:Nt-1).*dt
	τ_ind_range = get(kwargs,:τ_ind_range,1:Nt)
	t = vcat(0:div(Nt,2)-1,-div(Nt,2):-1).*dt
	t = t[τ_ind_range]

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)		
	δCt = -Ω_rot .* t .* ∂ϕ₂Ct(xobs1,xobs2,los;kwargs...)[τ_ind_range]
end 

# Linearized, with or without los, multiple 3D points
function δCt_uniform_rotation_rotatedwaves_linearapprox(xobs1::Point3D,
	xobs2_arr::Vector{<:Point3D},los::los_direction=los_radial();kwargs...)

	dν,Nt = read_parameters("dν","Nt";kwargs...)
	
	τ_ind_range = get(kwargs,:τ_ind_range,1:Nt)
	t = vcat(0:div(Nt,2)-1,-div(Nt,2):-1).*dt
	t = t[τ_ind_range]

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)
	
	δCt = -Ω_rot .* t .* ∂ϕ₂Ct(xobs1,xobs2_arr,los;kwargs...)
	δCt_arr = Vector{Float64}[]
	if !isnothing(τ_ind_range)
		for x2ind in 1:length(xobs2_arr)
			push!(δCt_arr,δCt[τ_ind_range[x2ind],x2ind])
		end
	else
		δCt_arr = δCt
	end
	return δCt_arr
end

# Linearized, with or without los, multiple 2D points
function δCt_uniform_rotation_rotatedwaves_linearapprox(nobs1::Point2D,
	nobs2_arr::Vector{<:Point2D},los::los_direction=los_radial();kwargs...)

	C = ∂ϕ₂Ct(nobs1,nobs2_arr,los;kwargs...)
	δCt_uniform_rotation_rotatedwaves_linearapprox(C;kwargs...)
end

# Linearized, with or without los, using precomputed time-domain cross-covariance
function δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_arr::Array{Float64};kwargs...)

	dν,Nt,dt = read_parameters("dν","Nt","dt";kwargs...)

	τ_ind_range = get(kwargs,:τ_ind_range,1:Nt)
	t = vcat(0:div(Nt,2)-1,-div(Nt,2):-1).*dt
	t = t[τ_ind_range]
	∂ϕ₂Ct_arr = ∂ϕ₂Ct_arr[τ_ind_range]

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)

	@. -Ω_rot * t * ∂ϕ₂Ct_arr
end

#######################################################################################################################################
# First Born approximation for sound speed perturbation
#######################################################################################################################################

function allocatearrays(::soundspeed,los::los_direction,obs_at_same_height)
	Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1,srcindG(los)...)
	drGsrc = zeros_Float64_to_ComplexF64(1:nr,0:0,srcindG(los)...)
	Gobs1 = zeros_Float64_to_ComplexF64(1:nr,0:1,srcindG(los)...)
	drGobs1 = zeros_Float64_to_ComplexF64(1:nr,0:0,srcindG(los)...)
	Gobs2,drGobs2 = Gobs1,drGobs1

	divGsrc = zeros(ComplexF64,nr,srcindG(los)...)
	divGobs = zeros(ComplexF64,nr,srcindG(los)...)
	
	# f_αjₒjₛ(r,rᵢ,rₛ) = -2ρc ∇⋅Gjₒ(r,rᵢ)_α ∇⋅Gjₛ(r,rₛ)_0
	fjₒjₛ_r₁_rsrc = zeros(ComplexF64,nr,srcindG(los)...)
	fjₒjₛ_r₂_rsrc = obs_at_same_height ? fjₒjₛ_r₁_rsrc : zero(fjₒjₛ_r₁_rsrc)

	# H_βαjₒjₛ(r;r₁,r₂,rₛ) = conj(f_αjₒjₛ(r,r₁,rₛ)) Gβ0jₛ(r₂,rₛ)
	Hjₒjₛω_r₁r₂ = zeros(ComplexF64,nr,obsindG(los)...,srcindG(los)...)
	Hjₒjₛω_r₂r₁ = obs_at_same_height ? Hjₒjₛω_r₁r₂ : zero(Hjₒjₛω_r₁r₂)

	@eponymtuple(Gsrc,drGsrc,Gobs1,drGobs1,Gobs2,
		drGobs2,divGsrc,divGobs,
		fjₒjₛ_r₁_rsrc,fjₒjₛ_r₂_rsrc,
		Hjₒjₛω_r₁r₂,Hjₒjₛω_r₂r₁)
end

function δCω_isotropicδc_firstborn_integrated_over_angle_partial(
	ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_radial,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,c_scale=1+1e-5,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=1)
	p_Gobs1 = read_all_parameters(p_Gobs1,r_src=xobs1.r,c_scale=1)
	p_Gobs2 = read_all_parameters(p_Gobs2,r_src=xobs2.r,c_scale=1)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	Gfn_path_obs1,NGfn_files_obs1 =  p_Gobs1.path,p_Gobs1.num_procs
	Gfn_path_obs2,NGfn_files_obs2 = p_Gobs2.path,p_Gobs2.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn,dω = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	δc_scale = c_scale-1
	δc = δc_scale.*c
		
	Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2 = 
		Gfn_fits_files((Gfn_path_src,Gfn_path_obs1,Gfn_path_obs2),
		(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc::ProductSplit,
		(NGfn_files_src,NGfn_files_obs1,NGfn_files_obs2))

	δC_rω = zeros(ComplexF64,nr,ν_ind_range)
	
	arr = allocatearrays(soundspeed(),los,r₁_ind == r₂_ind)
	@unpack Gsrc,drGsrc,divGsrc,Gobs1,drGobs1,Gobs2,divGobs = arr
	fjj_r₁_rsrc,fjj_r₂_rsrc = arr.fjₒjₛ_r₁_rsrc, arr.fjₒjₛ_r₂_rsrc
	H¹₁jj_r₁r₂,H¹₁jj_r₂r₁ = arr.Hjₒjₛω_r₁r₂, arr.Hjₒjₛω_r₂r₁

	lmax = maximum(ℓ_arr)
	Pl_cosχ = Pl(cosχn1n2(xobs1,xobs2),lmax=lmax)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
		
		ω = ω_arr[ω_ind]
		
		# Green function about source location
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
		(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:2,1,1)

		Gγr_r₁_rsrc = αrcomp(Gsrc,r₁_ind,los)
		Gγr_r₂_rsrc = αrcomp(Gsrc,r₂_ind,los)

		# Derivative of Green function about source location
		read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
		(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:1,1,2)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs1,:,1:2,1,1)

		# Derivative of Green function about receiver location
		read_Gfn_file_at_index!(drGobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs1,:,1:1,1,2)

		radial_fn_isotropic_δc_firstborn!(fjj_r₁_rsrc,
			Gsrc,drGsrc,divGsrc,Gobs1,drGobs1,divGobs,ℓ)

		Hjₒjₛω!(H¹₁jj_r₁r₂,fjj_r₁_rsrc,Gγr_r₂_rsrc)

		if r₁_ind != r₂_ind
			# Green function about receiver location
    		read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
    			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs2,:,1:2,1,1)

    		# Derivative of Green function about receiver location
    		read_Gfn_file_at_index!(drGobs2,Gfn_fits_files_obs2,
    			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs2,:,1,1,2)

			radial_fn_isotropic_δc_firstborn!(fjj_r₂_rsrc,
				Gsrc,drGsrc,divGsrc,Gobs2,drGobs2,divGobs,ℓ)

			Hjₒjₛω!(H¹₁jj_r₂r₁,fjj_r₂_rsrc,Gγr_r₁_rsrc)
		end

		pre = ω^2*Powspec(ω)* (2ℓ+1)/4π * Pl_cosχ[ℓ]
		iszero(pre) && continue
		@. δC_rω[:,ω_ind] += pre * (conj(H¹₁jj_r₂r₁) + H¹₁jj_r₁r₂)

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)
	closeGfnfits(Gfn_fits_files_obs1)
	closeGfnfits(Gfn_fits_files_obs2)

	C = parent(δC_rω)
	
	δCω = simps((@. r^2 * δc * C),r)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	δCω
end

function δCω_isotropicδc_firstborn_integrated_over_angle_partial(
	ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_earth,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,c_scale=1+1e-5,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=1)
	p_Gobs1 = read_all_parameters(p_Gobs1,r_src=xobs1.r,c_scale=1)
	p_Gobs2 = read_all_parameters(p_Gobs2,r_src=xobs2.r,c_scale=1)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	Gfn_path_obs1,NGfn_files_obs1 =  p_Gobs1.path,p_Gobs1.num_procs
	Gfn_path_obs2,NGfn_files_obs2 = p_Gobs2.path,p_Gobs2.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	ϵ = c_scale-1
		
	Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2 = 
		Gfn_fits_files((Gfn_path_src,Gfn_path_obs1,Gfn_path_obs2),
		(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,
		(NGfn_files_src,NGfn_files_obs1,NGfn_files_obs2))

	δC_rω = zeros(ComplexF64,nr,ν_ind_range)
	
	arrs = allocatearrays(soundspeed(),los,r₁_ind == r₂_ind)
	@unpack Gsrc,drGsrc,divGsrc,Gobs1,drGobs1,Gobs2,divGobs = arrs
	@unpack Hjₒjₛω_r₁r₂, Hjₒjₛω_r₂r₁ = arrs
	f_r₁_rsrc,f_r₂_rsrc = arrs.fjₒjₛ_r₁_rsrc, arrs.fjₒjₛ_r₂_rsrc

	# covariant components
	l1,l2 = line_of_sight_covariant(xobs1,xobs2,los)

	ℓ_range = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	Y12 = computeY₀₀(los,xobs1,xobs2,ℓ_range)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]
		pre = ω^2 * Powspec(ω) * (-1)^ℓ * √(2ℓ+1)

		# Green function about source location
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
		(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),1)

		Gγr_r₁_rsrc = αrcomp(Gsrc,r₁_ind,los)
		Gγr_r₂_rsrc = αrcomp(Gsrc,r₂_ind,los)

		# Derivative of Green function about source location
		read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
		(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:1,srcindFITS(los),2)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs1,:,1:2,srcindFITS(los),1)

		# Derivative of Green function about receiver location
		read_Gfn_file_at_index!(drGobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs1,:,1:1,srcindFITS(los),2)

		radial_fn_isotropic_δc_firstborn!(f_r₁_rsrc,
			Gsrc,drGsrc,divGsrc,Gobs1,drGobs1,divGobs,ℓ)

		Hjₒjₛω!(Hjₒjₛω_r₁r₂,f_r₁_rsrc,Gγr_r₂_rsrc)

		if r₁_ind != r₂_ind
			# Green function about receiver location
    		read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
    			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs2,:,1:2,srcindFITS(los),1)

    		# Derivative of Green function about receiver location
    		read_Gfn_file_at_index!(drGobs2,Gfn_fits_files_obs2,
    			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs2,:,1:1,srcindFITS(los),2)

			radial_fn_isotropic_δc_firstborn!(f_r₂_rsrc,
				Gsrc,drGsrc,divGsrc,Gobs2,drGobs2,divGobs,ℓ)

			Hjₒjₛω!(Hjₒjₛω_r₂r₁,f_r₂_rsrc,Gγr_r₁_rsrc)
		end

		Y12ℓ = Y12[ℓ]
		ind00 = modeindex(firstshmodes(Y12ℓ),(0,0))

		@inbounds for β in axes(Y12ℓ,2), α in axes(Y12ℓ,1)
			
			pre_llY = pre * l1[α] * l2[β] * Y12ℓ[α,β,ind00]
			iszero(pre_llY) && continue

			for r_ind in axes(δC_rω,1)
				δC_rω[r_ind,ω_ind] += pre_llY *
				( Hjₒjₛω_r₁r₂[r_ind,abs(α),abs(β)] + 
					conj(Hjₒjₛω_r₂r₁[r_ind,abs(β),abs(α)]) )
			end
		end

		signaltomaster!(progress_channel)
	end

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2))

	C = parent(δC_rω)
	δCω = simps((@. r^2 * (ϵ*c) * C),r)
	
	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	δCω
end

function _δCω_isotropicδc_firstborn(xobs1,xobs2,args...;kwargs...)
	r_src,r_obs,c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc,p_Gobs1,p_Gobs2 = read_parameters_for_points(xobs1,xobs2;kwargs...,c_scale=1)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	δCω_in_range = pmapsum_timed(
		δCω_isotropicδc_firstborn_integrated_over_angle_partial,
		modes_iter,xobs1,xobs2,args...,
		p_Gsrc,p_Gobs1,p_Gobs2,r_src,r_obs,c_scale;
		progress_str="Modes summed in δCω_c_FB : ")

	δCω_isotropicδc_FB = pad_zeros_ν(δCω_in_range,
					ν_ind_range,Nν_Gfn,ν_start_zeros,ν_end_zeros)
end

# Without los, radial components, 3D points
function δCω_isotropicδc_firstborn_integrated_over_angle(xobs1::Point3D,xobs2::Point3D,
	los::los_radial=los_radial();kwargs...)

	δCω_isotropicδc_FB = _δCω_isotropicδc_firstborn(xobs1,xobs2,los;kwargs...)
	@save_to_fits_and_return(δCω_isotropicδc_FB)
end

# With los, 3D points
function δCω_isotropicδc_firstborn_integrated_over_angle(xobs1::Point3D,xobs2::Point3D,
	los::los_earth;kwargs...)
	
	δCω_isotropicδc_FB_los = _δCω_isotropicδc_firstborn(xobs1,xobs2,los;kwargs...)
	@save_to_fits_and_return(δCω_isotropicδc_FB_los)
end

# 2D points
@two_points_on_the_surface δCω_isotropicδc_firstborn_integrated_over_angle

# 3D points, single frequency
function δCω_isotropicδc_firstborn_integrated_over_angle(xobs1,xobs2,ν::Real,
	los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack ν_arr,ν_start_zeros = p_Gsrc

	ν_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_range = ν_ind:ν_ind

	δCω_isotropicδc_firstborn_integrated_over_angle(
		xobs1,xobs2,los;
		kwargs...,ν_ind_range=ν_ind_range)[ν_ind+ν_start_zeros]
end

# Catch-all method to work with precomputed arrays etc
function δCt_isotropicδc_firstborn_integrated_over_angle(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	dν = read_parameters("dν";kwargs...)[1]

	δCω = δCω_isotropicδc_firstborn_integrated_over_angle(xobs1,xobs2,los;kwargs...)
	δCt_isotropicδc_FB = fft_ω_to_t(δCω,dν)

	@save_to_fits_and_return(δCt_isotropicδc_FB,los)
end

#######################################################################################################################################
# δC(ω) = C(ω) - C0(ω) for sound speed perturbations
#######################################################################################################################################

function δCω_isotropicδc_C_minus_C0_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_earth,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_G′src::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,c_scale=1+1e-5,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src,c_scale=1)
	p_G′src = read_all_parameters(p_G′src,r_src=r_src,c_scale=c_scale)

	Gfn_path_src_c0,NGfn_files_c0 = p_Gsrc.path,p_Gsrc.num_procs
	Gfn_path_src_c′,NGfn_files_c′ =  p_G′src.path,p_G′src.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	Gfn_fits_files_src_c0,Gfn_fits_files_src_c′ = 
	Gfn_fits_files((Gfn_path_src_c0,Gfn_path_src_c′),(ℓ_arr,1:Nν_Gfn),
		ℓ_ωind_iter_on_proc,(NGfn_files_c0,NGfn_files_c′))

	Cω = zeros(ComplexF64,ν_ind_range)

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	# covariant components
	l1,l2 = line_of_sight_covariant(xobs1,xobs2,los)

	G1_c′ = zeros_Float64_to_ComplexF64(0:1)
	G1_c0 = zeros_Float64_to_ComplexF64(0:1)
	if r₁_ind != r₂_ind
		G2_c′ = zeros_Float64_to_ComplexF64(0:1)
		G2_c0 = zeros_Float64_to_ComplexF64(0:1)
	else
		G2_c′ = G1_c′
		G2_c0 = G1_c0
	end

	ℓ_range = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	Y12 = computeY₀₀(los,xobs1,xobs2,ℓ_range)
	vinds = VSHvectorinds(xobs1,xobs2)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		# Green function about source location at observation point 1
		read_Gfn_file_at_index!(G1_c0,Gfn_fits_files_src_c0,
		(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_c0,r₁_ind,1:2,1,1)

		read_Gfn_file_at_index!(G1_c′,Gfn_fits_files_src_c′,
		(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_c′,r₁_ind,1:2,1,1)

		# Green function about source location at observation point 2
		if r₁_ind != r₂_ind
			read_Gfn_file_at_index!(G2_c0,Gfn_fits_files_src_c0,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_c0,r₂_ind,1:2,1,1)

			read_Gfn_file_at_index!(G2_c′,Gfn_fits_files_src_c′,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_c′,r₂_ind,1:2,1,1)
		end

		Y12ℓ = Y12[ℓ]

		@inbounds for β in vinds,α in vinds
			GG_c0 = conj(G1_c0[abs(α)])*G2_c0[abs(β)]
			GG_c′ = conj(G1_c′[abs(α)])*G2_c′[abs(β)]
			δGG = GG_c′ - GG_c0
			Cω[ω_ind] += ω^2 * Powspec(ω) * δGG * (-1)^ℓ * √(2ℓ+1) *
						l1[α] * l2[β] * Y12ℓ[α,β]
		end

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src_c′)
	closeGfnfits(Gfn_fits_files_src_c0)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	parent(Cω)
end

function _δCω_isotropicδc_C_minus_C0(xobs1,xobs2,
	los::los_earth,args...;kwargs...)
	r_src,r_obs,c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(r_src=r_src,c_scale=1)
	p_G′src = read_all_parameters(r_src=r_src,c_scale=c_scale)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	Cω_in_range = pmapsum_timed(
		δCω_isotropicδc_C_minus_C0_partial,
		modes_iter,xobs1,xobs2,los,args...,p_Gsrc,p_G′src,
		r_src,r_obs,c_scale;
		progress_str="Modes summed in δCω_c_CmC0 : ")

	δCω_isotropicδc_CmC0_los = pad_zeros_ν(
				Cω_in_range,ν_ind_range,Nν_Gfn,
				ν_start_zeros,ν_end_zeros)
end

# Without los, radial components, 3D points
function δCω_isotropicδc_C_minus_C0(xobs1::Point3D,xobs2::Point3D,
	los::los_radial=los_radial();kwargs...)

	c_scale = get(kwargs,:c_scale,1+1e-5)

	C′ = Cω(xobs1,xobs2,los;kwargs...,c_scale=c_scale)
	C0 = Cω(xobs1,xobs2,los;kwargs...,c_scale=1)

	δCω_isotropicδc_CmC0 = @. C′- C0
	@save_to_fits_and_return(δCω_isotropicδc_CmC0)
end

# With los, 3D points
function δCω_isotropicδc_C_minus_C0(xobs1::Point3D,xobs2::Point3D,los::los_earth;kwargs...)

	δCω_isotropicδc_CmC0_los = _δCω_isotropicδc_C_minus_C0(
								xobs1,xobs2,los;kwargs...)
	@save_to_fits_and_return(δCω_isotropicδc_CmC0_los)
end

# 2D points
@two_points_on_the_surface δCω_isotropicδc_C_minus_C0

# With or without los, 3D points, single frequency
function δCω_isotropicδc_C_minus_C0(xobs1,xobs2,ν::Real,
	los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...,c_scale=1)
	@unpack ν_arr,ν_start_zeros = p_Gsrc

	ν_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_range = ν_ind:ν_ind

	δCω_isotropicδc_C_minus_C0(xobs1,xobs2,los;
		kwargs...,ν_ind_range=ν_ind_range)[ν_ind+ν_start_zeros]
end

########################################################################################
# δC(t) = C(t) - C0(t) for sound speed perturbations
########################################################################################

# With or without los, 3D points
function δCt_isotropicδc_C_minus_C0(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	dν = read_parameters("dν";kwargs...)[1]

	δCω = δCω_isotropicδc_C_minus_C0(xobs1,xobs2,los;kwargs...)
	δCt_isotropicδc_CmC0 = fft_ω_to_t(δCω,dν)
	@save_to_fits_and_return(δCt_isotropicδc_CmC0,los)
end

# With or without los, precomputed arrays
function δCt_isotropicδc_C_minus_C0(δCω::AbstractArray;kwargs...)

	dν = read_parameters("dν";kwargs...)[1]

	δCt_isotropicδc_CmC0 = fft_ω_to_t(δCω,dν)
	@save_to_fits_and_return(δCt_isotropicδc_CmC0)
end

########################################################################################################################
# Window function
########################################################################################################################

function bounce_filter(Δϕ,n)
	nparams = 5
	coeffs = Dict()
	for i in [1,2,4]
		coeffs[i] = Dict("high"=>zeros(nparams),"low"=>zeros(nparams))
	end

	coeffs[1]["high"] = [2509.132334896018,12792.508296270391,-13946.527195127102,8153.75242742649,-1825.7567469552703]
	coeffs[1]["low"] = [40.821191938380714,11410.21390421857,-11116.305124138207,5254.244817703224,-895.0009393800744]

	coeffs[2]["high"] = [4083.6946001848364,14924.442447995087,-13880.238239469609,7562.499279468063,-1622.5318939228978]
	coeffs[2]["low"] = [2609.4406668522433,10536.81683213881,-7023.811081076518,2586.7238222832298,-348.9245124332354]

	coeffs[4]["high"] = [6523.103468645263,16081.024611219753,-7085.7174198723405,973.4990690666436,236.95568587146957]
	coeffs[4]["low"] = [5150.314633252216,15040.045600508669,-8813.047362534506,3878.5398150601663,-870.3633232120256]

	τ_low,τ_high = 0.,0.
	for (i,c) in enumerate(coeffs[n]["high"])
		τ_high += c*Δϕ^(i-1)
	end

	for (i,c) in enumerate(coeffs[n]["low"])
		τ_low += c*Δϕ^(i-1)
	end

	return τ_low,τ_high
end

function gaussian_fit(x,y)
	# use the fact that if y=Gaussian(x), log(y) = quadratic(x)
	# quadratic(x) = ax² + bx + c
	# Gaussian(x) = A*exp(-(x-x0)²/2σ²)
	# the parameters are σ=√(-1/2a), x0 = -b/2a, A=exp(c-b^2/4a)
	c,b,a=polyfit(x,log.(y),2).a
	A = exp(c-b^2/4a)
	x0 = -b/2a
	σ = √(-1/2a)
	return A,x0,σ
end

function time_window_bounce_filter(xobs1,xobs2,dt,bounce_no=1)
	time_window_bounce_filter(acos(cosχn1n2(xobs1,xobs2)),dt,bounce_no)
end

function time_window_bounce_filter(Δϕ::Real,dt,bounce_no=1)
	τ_low,τ_high = bounce_filter(Δϕ,bounce_no)
	τ_low_ind = floor(Int64,τ_low/dt); 
	τ_high_ind = ceil(Int64,τ_high/dt)
	return τ_low_ind,τ_high_ind
end

function time_window_indices_by_fitting_bounce_peak(C_t::AbstractVector{Float64},
	xobs1,xobs2;dt,bounce_no=1,kwargs...)

	τ_low_ind,τ_high_ind = time_window_bounce_filter(xobs1,xobs2,dt,bounce_no)
	time_window_indices_by_fitting_bounce_peak(C_t,τ_low_ind,τ_high_ind;dt=dt,kwargs...)
end

function time_window_indices_by_fitting_bounce_peak(C_t::AbstractVector{Float64},
	τ_low_ind::Int64,τ_high_ind::Int64;
	dt,Nt=size(C_t,1),kwargs...)
	
	env = abs.(hilbert(C_t[1:div(Nt,2)]))
	peak_center = argmax(env[τ_low_ind:τ_high_ind]) + τ_low_ind - 1
	points_around_max = env[peak_center-2:peak_center+2]
	amp = env[peak_center]
	A,t0,σt = gaussian_fit(peak_center-2:peak_center+2, points_around_max)

	t_inds_range = max(1,floor(Int64,t0 - 2σt)):min(ceil(Int64,t0 + 2σt),Nt)
end

function time_window_indices_by_fitting_bounce_peak(C_t::AbstractMatrix{Float64},
	xobs1,xobs2_arr::Vector,args...;kwargs...)

	t_inds_range = Vector{UnitRange}(undef,size(C_t,2))
	for (x2ind,xobs2) in enumerate(xobs2_arr)
		t_inds_range[x2ind] = time_window_indices_by_fitting_bounce_peak(C_t[:,x2ind],
								xobs1,xobs2;kwargs...)
	end
	return t_inds_range
end

function time_window(a::Vector,τ_ind_range)
	b = zeros(size(a))
	@. b[τ_ind_range] = 1
	return b
end

function time_window(a::Matrix{T},τ_ind_range::Vector) where {T}
	b = zeros(size(a))
	for (idx,τ_inds) in enumerate(τ_ind_range)
		@. b[τ_inds,idx] = 1
	end
	return b
end

function ht(::TravelTimes,Cω_x1x2::AbstractArray{ComplexF64},
	xobs1,xobs2;bounce_no=1,kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack ν_full,Nt,dt,dν = p_Gsrc

	ω_full = 2π.*ν_full

	C_t = fft_ω_to_t(Cω_x1x2,dν)
	∂tCt_ω = @. Cω_x1x2*im*ω_full
	∂tCt = fft_ω_to_t(∂tCt_ω,dν)

	τ_ind_range = get(kwargs,:τ_ind_range,nothing)
	if isnothing(τ_ind_range)
		τ_ind_range = time_window_indices_by_fitting_bounce_peak(C_t,xobs1,xobs2;
					dt=dt,Nt=Nt,bounce_no=bounce_no)
	end

	f_t = time_window(∂tCt,τ_ind_range)

	h_t =  (@. -f_t * ∂tCt) ./ sum((@. f_t*∂tCt^2 * dt),dims=1)
end

function ht(::Amplitudes,Cω_x1x2::AbstractArray{ComplexF64},
	xobs1,xobs2;bounce_no=1,kwargs...)

	ν_full,Nt,dt,dν = read_parameters("ν_full","Nt","dt","dν";kwargs...)

	ω_full = 2π.*ν_full

	C_t = fft_ω_to_t(Cω_x1x2,dν)

	τ_ind_range = get(kwargs,:τ_ind_range,nothing)
	if isnothing(τ_ind_range)
		τ_ind_range = time_window_indices_by_fitting_bounce_peak(C_t,xobs1,xobs2;
					dt=dt,Nt=Nt,bounce_no=bounce_no)
	end

	f_t = time_window(C_t,τ_ind_range)

	h_t =  (@. f_t * C_t) ./ sum((@. f_t*C_t^2 * dt),dims=1)
end

function ht(m::SeismicMeasurement,xobs1,xobs2,los::los_direction=los_radial();kwargs...)
	Cω_x1x2 = Cω(xobs1,xobs2,los;kwargs...)
	ht(m,Cω_x1x2,xobs1,xobs2;kwargs...)
end

function hω(m::SeismicMeasurement,args...;kwargs...)

	dt,ν_start_zeros,ν_arr = read_parameters("dt","ν_start_zeros","ν_arr";kwargs...)

	h_t = ht(m,args...;kwargs...)
	h_ω = fft_t_to_ω(h_t,dt)
	ν_ind_range = get(kwargs,:ν_ind_range,axes(ν_arr,1)) .+ ν_start_zeros

	OffsetArray(h_ω[ν_ind_range,..],axes(ν_ind_range,1),axes(h_ω)[2:end]...)
end

function hω(m::SeismicMeasurement,xobs1,xobs2,ν::Real,
	los::los_direction=los_radial();kwargs...)

	ν_arr = read_parameters("ν_arr";kwargs...)[1]
	ν_ind = searchsortedfirst(ν_arr,ν)
	hω(m,xobs1,xobs2,los;kwargs...)[ν_ind]
end

end # module

