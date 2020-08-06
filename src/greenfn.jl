#################################################################
# Green function radial components, main function
#################################################################
include("$(@__DIR__)/finite_difference.jl")
include("$(@__DIR__)/continuous_FFT.jl")
include("$(@__DIR__)/directions.jl")
include("$(@__DIR__)/timer_utils.jl")

module Greenfn_radial

using DelimitedFiles
using LinearAlgebra
using Reexport
using SparseArrays
using SuiteSparse

@reexport using ..Finite_difference
@reexport using ..Directions
@reexport using ..Continuous_FFT
@reexport using EllipsisNotation
@reexport using FileIO
@reexport using FITSIO
@reexport using JLD2
@reexport using OffsetArrays
@reexport using ParallelUtilities
@reexport using Parameters
@reexport using Polynomials
@reexport using Printf
@reexport using ProgressMeter
@reexport using WignerSymbols

# Choose scratch directory
const SCRATCH = get(ENV,"SCRATCH",
	isdir(joinpath("/scratch",ENV["USER"])) ? 
	joinpath("/scratch",ENV["USER"]) : pwd())

export Rsun
export nr
export r
export dr
export ddr
export c
export ρ
export g
export N2
export γ_damping
export r_src_default
export r_obs_default
export read_rsrc_robs_c_scale
export radial_grid_closest
export radial_grid_index
export components_radial_source
export all_components
export Gfn_path_from_source_radius
export Ω
export assign_Gfn_components!
export read_Gfn_file_at_index!
export read_Gfn_file_at_index
export Gfn_fits_files
export divG_radial
export radial_fn_isotropic_δc_firstborn!
export radial_fn_δc_firstborn!
export radial_fn_uniform_rotation_firstborn!
export Gfn_processor_range
export get_numprocs
export zeros_Float64_to_ComplexF64
export Njₒjₛs
export ζjₒjₛs
export Gⱼₒⱼₛω_u⁰⁺_firstborn!
export Hγₗⱼₒⱼₛω_α₁α₂_firstborn!
export Hjₒjₛω!

export read_parameters
export read_all_parameters
export ParamsGfn
export @unpack_all_parameters

export closeGfnfits

export SCRATCH

export srcindFITS
export srcindG
export obsindFITS
export obsindG
export αrcomp

################################################################################
# Utility functions
################################################################################

# Indexing

@inline srcindFITS(::los_radial) = 1
@inline srcindFITS(::los_earth) = 1:2

@inline srcindG(::los_radial) = ()
@inline srcindG(::los_earth) = (0:1,)

@inline obsindFITS(::los_radial) = 1
@inline obsindFITS(::los_earth) = 1:2

@inline obsindG(::los_radial) = ()
@inline obsindG(::los_earth) = (0:1,)

@inline αrcomp(G::AbstractArray{ComplexF64,2},r_ind,α) = G[r_ind,α]
@inline αrcomp(G::AbstractArray{ComplexF64,3},r_ind,α) = G[r_ind,α,0]
@inline αrcomp(G::AbstractArray{ComplexF64,2},r_ind,::los_radial) = αrcomp(G,r_ind,0)
@inline αrcomp(G::AbstractArray{ComplexF64,3},r_ind,::los_radial) = αrcomp(G,r_ind,0)
@inline function αrcomp(G::AbstractArray{ComplexF64,2},r_ind,los::los_earth)
	αrcomp(G,r_ind,Base.IdentityUnitRange(0:1))
end
@inline function αrcomp(G::AbstractArray{ComplexF64,3},r_ind,los::los_earth)
	αrcomp(G,r_ind,Base.IdentityUnitRange(0:1))
end

##################################################################################
# Compute the processor range for a particular range of modes
##################################################################################

function ℓ′ω_range_from_ℓω_range(ℓ_ωind_iter_on_proc::ProductSplit,s_max,ℓ_arr)
	ω_ind_min,ω_ind_max = extrema(ℓ_ωind_iter_on_proc,dim=2)
	ℓ′min_ωmin = minimum(minimum(intersect(ℓ_arr,max(ℓ-s_max,0):ℓ+s_max))
		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc if ω_ind==ω_ind_min)
	ℓ′max_ωmax = maximum(maximum(intersect(ℓ_arr,max(ℓ-s_max,0):ℓ+s_max))
		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc if ω_ind==ω_ind_max)
	return (ℓ′min_ωmin,ω_ind_min),(ℓ′max_ωmax,ω_ind_max)
end

function Gfn_processor_range((ℓ_arr,ω_ind_arr),ℓ_ωind_iter_on_proc::ProductSplit,s_max,num_procs)
	modes = ℓ′ω_range_from_ℓω_range(ℓ_ωind_iter_on_proc,s_max,ℓ_arr)
	proc_id_min = whichproc((ℓ_arr,ω_ind_arr),first(modes),num_procs)
	proc_id_max = whichproc((ℓ_arr,ω_ind_arr),last(modes),num_procs)
	return proc_id_min:proc_id_max
end

###################################################################################
## Read Green function slices (real and imaginary parts) and assign to arrays
###################################################################################

function zeros_Float64_to_ComplexF64(inds::Vararg{<:AbstractUnitRange})
	sz = map(length,inds)
	OffsetArray(dropdims(reinterpret(ComplexF64,zeros(2,sz...)),dims=1),inds...)
end

const ReshapedArrayRealToComplex{C,N,R,M} = 
		OffsetArray{C,N,
		Base.ReshapedArray{C,N,
		Base.ReinterpretArray{C,M,R,Array{R,M}},Tuple{}}}

parentarray(a::ReshapedArrayRealToComplex) = parentarray(parent(a))
parentarray(a::Base.ReshapedArray) = parentarray(parent(a))
parentarray(a::Base.ReinterpretArray) = parentarray(parent(a))
parentarray(a::Array) = a

function assign_Gfn_components!(G_Re_Im::Array{Float64},G::OffsetArray{ComplexF64})
	ax_offset = CartesianIndices(axes(G))
	ax_flat = CartesianIndices(axes(G_Re_Im)[2:end])
	assign_Gfn_components!(G_Re_Im,G,ax_flat,ax_offset)
end

function assign_Gfn_components!(G_Re_Im::Array{Float64},G::OffsetArray{ComplexF64},
	ax_flat,ax_offset)

	@inbounds for (ind_offset,ind_flat) in zip(ax_offset,ax_flat)
		G[ind_offset] = G_Re_Im[1,ind_flat] + im*G_Re_Im[2,ind_flat]
	end
end

function read_Gfn_file_at_index!(G::AbstractArray{<:Complex},
	G_fits_dict::Dict,(ℓ_arr,ω_ind_arr)::Tuple{Vararg{AbstractUnitRange,2}},
	ℓω::NTuple{2,Int},num_procs::Integer,I::Vararg{Any,4})

	proc_id_mode,ℓω_index_in_file = whichproc_localindex((ℓ_arr,ω_ind_arr),ℓω,num_procs)
	G_file = G_fits_dict[proc_id_mode].hdu
	read_Gfn_file_at_index!(G,G_file,ℓω_index_in_file,I...)
end

function check_if_dims_are_ok(M,N)
	if M != N+1
		throw(DimensionMismatch("The real array should have one "*
			"fewer dimension than the complex one"))
	end
end

@inline function read_Gfn_file_at_index!(G::ReshapedArrayRealToComplex{<:Complex,N,<:Real,M},
	G_fits::FITS,ℓω_index_in_file::Integer,I::Vararg{Any,4}) where {M,N}

	G_hdu = G_fits[1]
	read_Gfn_file_at_index!(G,G_hdu,ℓω_index_in_file,I...)
end

function read_Gfn_file_at_index!(G::ReshapedArrayRealToComplex{<:Complex,N,<:Real,M},
	G_hdu::ImageHDU,ℓω_index_in_file::Integer,I::Vararg{Any,4}) where {M,N}

	check_if_dims_are_ok(M,N)
	read!(G_hdu,parentarray(G),:,I...,ℓω_index_in_file)
	nothing
end

@inline function read_Gfn_file_at_index!(G::AbstractArray,
	G_file::FITS,ℓω_index_in_file::Integer,I::Vararg{Any,4})

	read_Gfn_file_at_index!(G,G_file[1],ℓω_index_in_file,I...)
end

function read_Gfn_file_at_index!(G::AbstractArray,
	G_file::ImageHDU,ℓω_index_in_file::Integer,I::Vararg{Any,4})

	assign_Gfn_components!(read(G_file,:,I...,ℓω_index_in_file),G)
end

function read_Gfn_file_at_index(G_fits_dict::Dict,
	(ℓ_arr,ω_ind_arr)::Tuple{Vararg{AbstractUnitRange,2}},
	ℓω::Tuple,num_procs::Integer,I::Vararg{Any,4})

	proc_id_mode,ℓω_index_in_file = whichproc_localindex((ℓ_arr,ω_ind_arr),ℓω,num_procs)
	G_file = G_fits_dict[proc_id_mode].hdu
	read_Gfn_file_at_index(G_file,ℓω_index_in_file,I...)
end

function read_Gfn_file_at_index(G_file::Union{FITS,ImageHDU},
	(ℓ_arr,ω_ind_arr)::Tuple{Vararg{AbstractUnitRange,2}},
	ℓω::NTuple{2,Int},num_procs::Integer,I::Vararg{Any,4})

	_,ℓω_index_in_file = whichproc_localindex((ℓ_arr,ω_ind_arr),ℓω,num_procs)
	read_Gfn_file_at_index(G_file,ℓω_index_in_file,I...)
end

@inline function read_Gfn_file_at_index(G_file::FITS,
	ℓω_index_in_file::Integer,I::Vararg{Any,4})

	G_hdu = G_file[1]
	read_Gfn_file_at_index(G_hdu,ℓω_index_in_file,I...)
end

function read_Gfn_file_at_index(G_hdu::ImageHDU,
	ℓω_index_in_file::Integer,I::Vararg{Any,4})

	sz = size(G_hdu) :: NTuple{6,Int}
	M = length(FITSIO._index_shape(sz,:,I...,ℓω_index_in_file))
	N = M-1

	Tparent = Array{Float64,M}

	Tret = Base.ReshapedArray{ComplexF64,N,
			Base.ReinterpretArray{ComplexF64,M,Float64,Tparent},Tuple{}}

	G = read(G_hdu,:,I...,ℓω_index_in_file) :: Tparent
	dropdims(reinterpret(ComplexF64,G),dims=1) :: Tret
end

function Gfn_fits_files(path::String,proc_id_range::AbstractUnitRange)
	d = Dict{Int64,NamedTuple{(:fits, :hdu),Tuple{FITS,ImageHDU}}}()
	for procid in proc_id_range
		filename = @sprintf "Gfn_proc_%03d.fits" procid
		filepath = joinpath(path,filename)
		f_FITS = FITS(filepath,"r")
		f_HDU = f_FITS[1]
		d[procid] = (fits=f_FITS,hdu=f_HDU)
	end
	d
end

function Gfn_fits_files(path::String,(ℓ_arr,ω_ind_arr),
	ℓ_ωind_iter_on_proc::ProductSplit,num_procs::Integer)
	proc_range = procrange_recast((ℓ_arr,ω_ind_arr),ℓ_ωind_iter_on_proc,num_procs)
	Gfn_fits_files(path,proc_range)
end

function Gfn_fits_files(path::String,(ℓ_arr,ω_ind_arr),ℓ_ωind_iter_on_proc::ProductSplit,
	s_max::Integer,num_procs::Integer)

	proc_range = Gfn_processor_range((ℓ_arr,ω_ind_arr),ℓ_ωind_iter_on_proc,s_max,num_procs)
	Gfn_fits_files(path,proc_range)
end

function Gfn_fits_files(paths::Union{AbstractVector,Tuple},
	(ℓ_arr,ω_ind_arr),
	ℓ_ωind_iter_on_proc,
	num_procs::Union{AbstractVector,Tuple})

	G_files = Vector{Dict{Int64,NamedTuple{(:fits,:hdu),Tuple{FITS,ImageHDU}}}}(
					undef,length(paths))
	for (ind,(path,np)) in enumerate(zip(paths,num_procs))
		G_files[ind] = Gfn_fits_files(path,(ℓ_arr,ω_ind_arr),ℓ_ωind_iter_on_proc,np)
	end
	return G_files
end

function closeGfnfits(d::Dict)
	map(x->close(x.fits),values(d))
end

# Directory to save output to

function Gfn_path_from_source_radius(r_src::Real;c_scale=1)
	dir = "Greenfn_src$((r_src/Rsun > 0.99 ? 
			(@sprintf "%dkm" (r_src-Rsun)/1e5) : 
			(@sprintf "%.2fRsun" r_src/Rsun) ))"
	if c_scale != 1
		dir *= "_c_scale_$(@sprintf "%g" c_scale)"
	end
	path= joinpath(SCRATCH,dir)
end

#####################################################################################
# Solve for components
#####################################################################################

function load_solar_model()
	
	modelS_meta = readdlm("$(@__DIR__)/ModelS.meta",comments=true, comment_char='#');
	Msun,Rsun = modelS_meta[1:2];

	modelS = readdlm("$(@__DIR__)/ModelS",comments=true, comment_char='#');
	modelS_detailed = readdlm("$(@__DIR__)/ModelS.detailed",comments=true, comment_char='#');

	HMIfreq=readdlm("$(@__DIR__)/m181q.1216");
	ℓ_HMI,ν_HMI,γ_HMI = HMIfreq[:,1],HMIfreq[:,3],HMIfreq[:,5];

	# Fit modes above ℓ=11 and 2mHz<ν<4mHz 
	mode_filter = (ℓ_HMI.>11) .& (ν_HMI .> 1.5e3) .& (ν_HMI .< 4.5e3);
	
	ν_HMI = ν_HMI[mode_filter];
	ℓ_HMI = ℓ_HMI[mode_filter];
	γ_HMI = γ_HMI[mode_filter];

	# Fit γ(ω) in Hz, the HMI frequencies are in μHz
	γ_damping = polyfit(ν_HMI.*(2π*1e-6),γ_HMI.*(2π*1e-6),3);
	# γ_damping(ω) = 2π*4e-6 # constant damping to compare with krishnendu
	
	# invert the grid to go from inside to outside, 
	# and leave out the center to avoid singularities
	# Start from r=0.2Rsun to compare with Mandal. et al 2017 
	
	r_start_ind_mandal = searchsortedfirst(modelS[:,1],0.2,rev=true)
	r_start_ind_skipzero = searchsortedfirst(modelS[:,1],2e-2,rev=true)
	r_start_ind_somewhere_inside = searchsortedfirst(modelS[:,1],0.5,rev=true)

	#set r_start_ind to r_start_ind_mandal to compare with Mandal. et al 2017 
	r_start_ind = r_start_ind_skipzero

	r = modelS[r_start_ind:-1:1,1]*Rsun; 

	nr = length(r);

	dr = D(nr,Dict(7=>3,43=>5))*r;
	ddr = dbydr(dr);
	c = modelS[r_start_ind:-1:1,2];
	ρ = modelS[r_start_ind:-1:1,3];

	G = 6.67428e-8 # cgs units
	m = @. Msun*exp(modelS_detailed[r_start_ind:-1:1,2])

	g = @. G*m/r^2

	N2 = @. g * modelS_detailed[r_start_ind:-1:1,15] / r

	return Rsun,nr,r,dr,ddr,c,ρ,g,N2,γ_damping
end

const Rsun,nr,r,dr,ddr,c,ρ,g,N2,γ_damping = load_solar_model()

radial_grid_index(r_pt::Real) = searchsortedfirst(r,r_pt)
radial_grid_closest(r_pt::Real) = r[searchsortedfirst(r,r_pt)]

const r_src_default = radial_grid_closest(Rsun - 75e5)
const r_obs_default = radial_grid_closest(Rsun + 200e5)

function read_rsrc_robs_c_scale(kwargs)
	r_src = get(kwargs,:r_src,r_src_default)
	r_obs = get(kwargs,:r_obs,r_obs_default)
	c_scale = get(kwargs,:c_scale,1)
	r_src,r_obs,c_scale
end

# Source components in Hansen VSH basis
function σsrc_grid(r_src=r_src_default)
	r_src_ind = radial_grid_index(r_src)
	r_src_on_grid = r[r_src_ind];
	σsrc = max(r_src_on_grid - r[max(1,r_src_ind-2)],r[min(nr,r_src_ind+2)] - r_src_on_grid)
end

function delta_fn_gaussian_approx(r_src,σsrc)
	@. exp(-(r - r_src)^2 / 2σsrc^2) / √(2π*σsrc^2) /r^2;
end

function source(ω,ℓ;r_src=r_src_default)

	r_src_on_grid = radial_grid_closest(r_src)
	σsrc = σsrc_grid(r_src_on_grid)

	delta_fn = delta_fn_gaussian_approx(r_src_on_grid,σsrc)

	Sr = append!(delta_fn[2:nr-1],zeros(nr))
	Sh_bottom = @. -√(ℓ*(ℓ+1))/ω^2 * delta_fn /(r*ρ)

	Sh = append!(zeros(nr),Sh_bottom[2:nr-1])

	return Sr,Sh;
end

# Functions that provide a sparse representation of the wave operator

function ℒr(;c_scale=1)

	c′=c.*c_scale
	N2′=@. N2+g^2/c^2*(1-1/c_scale^2)

	# Left edge ghost to use point β(r_in) in computing derivatives
	# Dirichlet condition β(r_out)=0 on the right edge
	L14 = dbydr(dr[2:nr-1],
				left_edge_ghost=true,left_edge_npts=3,
				right_edge_ghost=false,right_edge_npts=3,
				right_Dirichlet=true) # (nr-2 x nr-1)

	L14[diagind(L14,1)] .+= (@. g/c′^2)[2:nr-1]

	# Boundary condition on α(r_out)
	L22 = derivStencil(1,2,0,gridspacing=dr[end]) .+ sparsevec([3],[2/r[end] - g[end]/c′[end]^2],3) # (2 x 1)

	# Boundary condition on β(r_in)
	L33 = derivStencil(1,0,2,gridspacing=dr[1]) .+ sparsevec([1],[g[1]/c′[1]^2],3) # (2 x 1)

	# Right edge ghost to use point α(r_out) in computing derivatives
	# Dirichlet condition α(r_in)=0 on the left edge
	L41 = dbydr(dr[2:end-1],
				left_edge_ghost=false,left_edge_npts=3,
				right_edge_ghost=true,right_edge_npts=3,
				left_Dirichlet=true) # (nr-2 x nr-1)
	L41[diagind(L41,0)] .+= (@. 2/r - g/c′^2)[2:end-1]

	M = spzeros(ComplexF64,2*(nr-1),2*(nr-1))

	M[diagind(M)[1:nr-2]] = @. (ρ*N2′)[2:nr-1]

	M[1:nr-2,end-(nr-1)+1:end] = L14

	M[nr-1,nr-3:nr-1] = L22

	M[nr,nr:nr+2] = L33

	M[nr+1:end,1:nr-1] = L41

	M[diagind(M)[nr+1:end]] = @. (1/(ρ*c′^2)[2:nr-1]) # Assign to diagonals

	return M
end

Ω(ℓ,N) = √((ℓ+N)*(ℓ-N+1)/2)
ζjₒjₛs(j₁,j₂,ℓ) = (Ω(j₁,0)^2 + Ω(j₂,0)^2 - Ω(ℓ,0)^2)/(Ω(j₁,0)*Ω(j₂,0))
@inline Njₒjₛs(jₒ,jₛ,s) = √((2jₒ+1)*(2jₛ+1)/(4π*(2s+1)))

function ℒωℓr(ω,ℓ;c_scale=1)

	ω -= im*γ_damping(ω)
	
	M_ωℓ = ℒr(c_scale=c_scale)

	M_ωℓ[diagind(M_ωℓ)[1:nr-2]] .+= @. -ω^2 * ρ[2:end-1]
	M_ωℓ[diagind(M_ωℓ)[nr+1:end]] .+= @. -ℓ*(ℓ+1)/(ω^2 * (ρ*r^2)[2:end-1])

	dropzeros(M_ωℓ)
end

# Functions to write the frequency grid and other parameters to the data directory

function frequency_grid(Gfn_save_directory;kwargs...)
	ν_low=get(kwargs,:ν_low,2.0e-3)
	ν_high=get(kwargs,:ν_high,4.5e-3)
	num_ν=get(kwargs,:num_ν,1250)
	@assert(num_ν>1,"Need at least two points in frequency to construct the grid")
	ν_Nyquist=get(kwargs,:ν_Nyquist,16e-3)

	ℓ_arr = get(kwargs,:ℓ_arr,1:1)
	
	dν = (ν_high - ν_low)/(num_ν-1); dω = 2π*dν
	
	# choose values on a grid
	ν_low_index = Int64(floor(ν_low/dν)); ν_low = ν_low_index*dν
	ν_high_index = num_ν + ν_low_index - 1; ν_high = ν_high_index*dν;
	Nν_Gfn = num_ν
	ν_Nyquist_index = Int64(ceil(ν_Nyquist/dν)); ν_Nyquist = ν_Nyquist_index*dν
	
	Nν = ν_Nyquist_index + 1; Nt = 2*(Nν-1)
	ν_full = (0:ν_Nyquist_index).*dν;
	ν_arr = (ν_low_index:ν_high_index).*dν ;
	T=1/dν; dt = T/Nt;
	ν_start_zeros = ν_low_index # index starts from zero
	ν_end_zeros = ν_Nyquist_index - ν_high_index

	ω_arr = 2π .* ν_arr;

	if !isdir(Gfn_save_directory)
		mkdir(Gfn_save_directory)
	end
	@save(joinpath(Gfn_save_directory,"parameters.jld2"),
		ν_arr,ω_arr,ν_full,dν,dω,ℓ_arr,
		ν_start_zeros,ν_end_zeros,Nν,Nt,dt,T,Nν_Gfn,ν_Nyquist)
	ℓ_arr,ω_arr
end

function append_parameters(Gfn_save_directory;kwargs...)
	paramfile = joinpath(Gfn_save_directory,"parameters.jld2")
	params = jldopen(paramfile,"a+")
	for (k,v) in Dict(kwargs)
		params[string(k)] = v
	end
	close(params)
end

function update_parameters(Gfn_save_directory;kwargs...)
	paramfile = joinpath(Gfn_save_directory,"parameters.jld2")
	params = load(paramfile)
	for (k,v) in Dict(kwargs)
		params[string(k)] = v
	end
	save(paramfile,params)
end

struct ParamsGfn
	path::String
	Nν_Gfn::Int
	ν_Nyquist::Float64
	ℓ_arr::UnitRange{Int}
	dν::Float64
	dω::Float64
	Nν::Int
	ν_full::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
	ν_arr::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
	T::Float64
	dt::Float64
	Nt::Int
	ν_start_zeros::Int
	ν_end_zeros::Int
	ω_arr::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
	num_procs::Int
end

function read_all_parameters(;kwargs...)
	r_src = get(kwargs,:r_src,r_src_default)
	c_scale=get(kwargs,:c_scale,1)
	Gfn_path_src = Gfn_path_from_source_radius(r_src,c_scale=c_scale)
	# load all parameters from the file at the path
	params_all = load(joinpath(Gfn_path_src,"parameters.jld2"))
	# pack the parameters including the path into a struct
	ParamsGfn(Gfn_path_src,
		[params_all[String(k)] for k in fieldnames(ParamsGfn) if k != :path]...)
end

@inline read_all_parameters(p::ParamsGfn;kwargs...) = p
function read_all_parameters(::Nothing;kwargs...)
	read_all_parameters(;kwargs...) :: ParamsGfn
end

function read_parameters(params...;kwargs...)
	isempty(params) && return ()
	params_all = read_all_parameters(;kwargs...)
	Tuple(getproperty(params_all,Symbol(p)) for p in params)
end

get_numprocs(path) = load(joinpath(path,"parameters.jld2"),"num_procs")

# Functions to compute Green function components in the helicity basis
# We solve in the Hansen VSH basis first and change over to the helicity basis

function solve_for_components!(M,S,α,β)
	
	H = M\S # solve the equation

	@. α[2:nr] = H[1:nr-1]
	@. β[1:nr-1] = H[nr:end]
end

function compute_Gfn_components_onemode!(ω,ℓ,αrℓω,βrℓω,αhℓω,βhℓω;kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	tangential_source = get(kwargs,:tangential_source,true)
	c_scale = get(kwargs,:c_scale,1)

	Sr,Sh = source(ω,ℓ,r_src=r_src)

	M = ℒωℓr(ω,ℓ,c_scale=c_scale);

	M_lu = lu(M)

	# radial source
	solve_for_components!(M_lu,Sr,αrℓω,βrℓω)

	if tangential_source
		solve_for_components!(M_lu,Sh,αhℓω,βhℓω)
	end
end

compute_Gfn_components_onemode!(ω,ℓ,αrℓω,βrℓω,::Nothing,::Nothing;kwargs...) = 
compute_Gfn_components_onemode!(ω,ℓ,αrℓω,βrℓω;kwargs...)

function compute_Gfn_components_onemode!(ω,ℓ,αrℓω,βrℓω;kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	c_scale = get(kwargs,:c_scale,1)

	Sr, = source(ω,ℓ,r_src=r_src)

	M = ℒωℓr(ω,ℓ,c_scale=c_scale);
	M_lu = lu(M)

	# radial source
	solve_for_components!(M_lu,Sr,αrℓω,βrℓω)
end

function compute_G_somemodes_serial_oneproc(ℓ_ωind_proc::ProductSplit,
	r_src,c_scale,
	ω_arr,Gfn_save_directory,
	tracker::Union{Nothing,RemoteChannel}=nothing;
	tangential_source::Bool=true)
	
	save_path = joinpath(Gfn_save_directory,
						@sprintf "Gfn_proc_%03d.fits" workerrank())

	FITS(save_path,"w") do file

		# save real and imaginary parts separately
		β = 0:1
		γ = tangential_source ? (0:1) : (0:0)
		G = zeros(2,β,γ,nr,length(ℓ_ωind_proc),0:1)

		r_src_on_grid = radial_grid_closest(r_src)
		σsrc = σsrc_grid(r_src_on_grid)
		delta_fn = delta_fn_gaussian_approx(r_src_on_grid,σsrc)

		αr,βr = (zeros(ComplexF64,nr) for i in 1:2);
		αh,βh = tangential_source ? (zeros(ComplexF64,nr) for i=1:2) : (nothing,nothing)
		T = zeros(ComplexF64,1)

		for (ℓωind,(ℓ,ω_ind)) in enumerate(ℓ_ωind_proc)

			ω = ω_arr[ω_ind]

			compute_Gfn_components_onemode!(ω,ℓ,αr,βr,αh,βh,
				r_src=r_src,tangential_source=tangential_source,
				c_scale=c_scale);

			# radial component for radial source
			@inbounds for r_ind in eachindex(r)
				@. T = αr[r_ind]
				G[:,0,0,r_ind,ℓωind,0] .= reinterpret(Float64,T)

				# tangential component for radial source

				@. T = Ω(ℓ,0)/(ρ[r_ind]*r[r_ind]*ω^2) * βr[r_ind]
				G[:,1,0,r_ind,ℓωind,0] .= reinterpret(Float64,T)

				if tangential_source
					# radial component for tangential source
					@. T = αh[r_ind]/√2
					G[:,0,1,r_ind,ℓωind,0] .= reinterpret(Float64,T)
					
					# tangential component for tangential source
					@. T = Ω(ℓ,0)/(ρ[r_ind]*r[r_ind]*ω^2) * (βh[r_ind] - delta_fn[r_ind])
					G[:,1,1,r_ind,ℓωind,0] .= reinterpret(Float64,T)
				end
			end

			(tracker isa RemoteChannel) && put!(tracker,true)
		end

		# convert to (r,deriv,reim,β,γ,ℓω) from (reim,β,γ,r,ℓω,deriv)
		G = permutedims(G,[4,6,1,2,3,5])
		# derivatives

		inds = CartesianIndices(axes(G)[3:end])
		@inbounds for i in inds
			G[:,1,i] .= ddr * (@views G[:,0,i])
		end

		(tracker isa RemoteChannel) && put!(tracker,true)
		
		# convert to (reim,r,β,γ,deriv,ℓω) from (r,deriv,reim,β,γ,ℓω)
		G = permutedims(G,[3,1,4,5,2,6])

		write(file,parent(G))

		(tracker isa RemoteChannel) && put!(tracker,true)

	end # close file
	
	nothing
end

function compute_components_allmodes(r_src=r_src_default;kwargs...)

	tangential_source = get(kwargs,:tangential_source,true)
	c_scale=get(kwargs,:c_scale,1)
	Gfn_save_directory = Gfn_path_from_source_radius(r_src,c_scale=c_scale)

	if !isdir(Gfn_save_directory)
		mkdir(Gfn_save_directory)
	end

	println("Saving output to $Gfn_save_directory")

	ℓ_arr,ω_arr = frequency_grid(Gfn_save_directory;kwargs...);
	Nν_Gfn = length(ω_arr); ω_ind_arr = 1:Nν_Gfn

	modes_iter = Iterators.product(ℓ_arr,ω_ind_arr)
	num_tasks = length(modes_iter)
	num_procs = ParallelUtilities.nworkersactive(modes_iter.iterators)

	append_parameters(Gfn_save_directory,num_procs=num_procs)

	N_prog_ticks = num_tasks + 2num_procs

	tracker = RemoteChannel(()->Channel{Bool}(N_prog_ticks))
	prog_bar = Progress(N_prog_ticks,2,"Green functions computed : ")

	wrapper(x) = compute_G_somemodes_serial_oneproc(x,r_src,c_scale,ω_arr,
	Gfn_save_directory,tracker;tangential_source=tangential_source)

	# Useful for debugging
	# pmapbatch(wrapper,(ℓ_arr,ω_ind_arr));

	@sync begin
		@async begin
			try
				pmapbatch(wrapper,modes_iter);
				println("Finished computing Green functions")
			catch e
				rethrow()
			finally
				put!(tracker,false)
				finish!(prog_bar)
			end
		end
		@async while take!(tracker)
			 next!(prog_bar)
		end
	end;
	nothing # to suppress the task done message
end

function Gfn_reciprocity(;kwargs...)
	r_src = get(kwargs,:r_src,r_src_default)
	r_src_ind = radial_grid_index(r_src)
	r_obs = get(kwargs,:r_obs,r_obs_default)
	r_obs_ind = radial_grid_index(r_obs)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	Gfn_path_obs = Gfn_path_from_source_radius(r_obs)
	@load(joinpath(Gfn_path_src,"parameters.jld2"),
		ν_arr,Nν_Gfn,ℓ_arr,num_procs)

	num_procs_obs = get_numprocs(Gfn_path_obs)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)

	function summodes(ℓ_ωind_iter_on_proc)
		G_reciprocity = zeros(2,ℓ_range,ν_ind_range)

		Gfn_fits_files_src,
		Gfn_fits_files_obs = Gfn_fits_files(
							(Gfn_path_src,Gfn_path_obs),ℓ_arr,1:Nν_Gfn,
							ℓ_ωind_iter_on_proc,(num_procs,num_procs_obs))

		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

			G10_obs_src = read_Gfn_file_at_index(
				Gfn_fits_files_src,ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,
				r_obs_ind,2,1,1)

			G01_src_obs = read_Gfn_file_at_index(
				Gfn_fits_files_obs,ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs,
				r_src_ind,1,2,1)

			G_reciprocity[1,ℓ,ω_ind] = abs(G10_obs_src)
			G_reciprocity[2,ℓ,ω_ind] = abs(G01_src_obs)

		end

		return G_reciprocity
	end

	G_reciprocity = @fetchfrom workers()[1] permutedims(
					pmapsum(summodes,modes_iter),[3,2,1])
end

function divG_radial!(divG::AbstractArray{ComplexF64,1},ℓ::Integer,
	G::AbstractArray{ComplexF64,2},drG::AbstractArray{ComplexF64,2})

	# components in PB VSH basis
	pre = -2Ω(ℓ,0)

	@inbounds for r_ind in axes(divG,1)
		divG[r_ind] = pre * G[r_ind,1]/r[r_ind] + 
			drG[r_ind,0] + 2/r[r_ind]*G[r_ind,0]
	end

	divG
end

function divG_radial!(divG::AbstractArray{ComplexF64,2},ℓ::Integer,
	G::AbstractArray{ComplexF64,3},drG::AbstractArray{ComplexF64,3})

	# components in PB VSH basis
	pre = -2Ω(ℓ,0)

	@inbounds for β in axes(G,3), r_ind in axes(divG,1)
		divG[r_ind,β] = pre * G[r_ind,1,β]/r[r_ind] + 
		drG[r_ind,0,β] + 2/r[r_ind]*G[r_ind,0,β]
	end

	divG
end

# Radial components of dG for sound-speed perturbations
function radial_fn_δc_firstborn!(f::AbstractArray{ComplexF64,1},
	divGsrc::AbstractArray{ComplexF64,1},divGobs::AbstractArray{ComplexF64,1})

	@. f = -ρ*2c*divGobs*divGsrc
end

function radial_fn_δc_firstborn!(f::AbstractArray{ComplexF64,2},
	divGsrc::AbstractArray{ComplexF64,1},divGobs::AbstractArray{ComplexF64,2})
	
	@. f = -ρ*2c*divGobs*divGsrc
end

function radial_fn_isotropic_δc_firstborn!(f,Gsrc::AA,drGsrc::AA,divGsrc,
	Gobs::AA,drGobs::AA,divGobs,ℓ) where {AA<:AbstractArray{ComplexF64}}
	radial_fn_δc_firstborn!(f,Gsrc,drGsrc,ℓ,divGsrc,Gobs,drGobs,ℓ,divGobs)
end

# Only radial component
function radial_fn_δc_firstborn!(f::AbstractVector{<:Complex},
	Gsrc::AA,drGsrc::AA,ℓ::Integer,divGsrc::BB,
	Gobs::AA,drGobs::AA,ℓ′::Integer,divGobs::BB) where 
		{AA<:AbstractArray{<:Complex,2},BB<:AbstractArray{<:Complex,1}}

	# G is a tensor with one trailing vector index, the first axis is r
	# In this case the divG arrays are 1D (r)
	divG_radial!(divGobs,ℓ′,Gobs,drGobs)
	divG_radial!(divGsrc,ℓ,Gsrc,drGsrc)

	radial_fn_δc_firstborn!(f,divGsrc,divGobs)
end

# All components
function radial_fn_δc_firstborn!(f::AbstractMatrix{<:Complex},
	Gsrc::AA,drGsrc::AA,ℓ::Integer,divGsrc::BB,
	Gobs::AA,drGobs::AA,ℓ′::Integer,divGobs::BB) where 
		{AA<:AbstractArray{<:Complex,3},BB<:AbstractArray{<:Complex,2}}

	# G is a tensor with two vector indices
	# In this case the divG arrays are 2D (r,vec_ind)
	divG_radial!(divGobs,ℓ′,Gobs,drGobs)
	divG_radial!(divGsrc,ℓ,Gsrc,drGsrc)

	divGsrc_0 = view(divGsrc,:,0) # Source is radial

	radial_fn_δc_firstborn!(f,divGsrc_0,divGobs)
end

# All components
# H_βαjₒjₛ(r;r₁,r₂,rₛ) = conj(f_αjₒjₛ(r,r₁,rₛ)) Gβ0jₛ(r₂,rₛ)
function Hjₒjₛω!(Hjₒjₛω_r₁r₂::AbstractArray{ComplexF64,3},
	fjₒjₛ_r₁_rsrc::AbstractMatrix{ComplexF64},Grjₛ_r₂_rsrc::AbstractVector{ComplexF64})

	for γ in axes(Grjₛ_r₂_rsrc,1)
		Gγrjₛ_r₂_rsrc = Grjₛ_r₂_rsrc[γ]
		for α in axes(fjₒjₛ_r₁_rsrc,2), r_ind in axes(fjₒjₛ_r₁_rsrc,1)
			Hjₒjₛω_r₁r₂[r_ind,α,γ] = conj(fjₒjₛ_r₁_rsrc[r_ind,α]) * Gγrjₛ_r₂_rsrc
		end
	end
end

# Only radial component
# H_00jₒjₛ(r;r₁,r₂,rₛ) = conj(f_0jₒjₛ(r,r₁,rₛ)) G00jₛ(r₂,rₛ)
function Hjₒjₛω!(Hjₒjₛω_r₁r₂::Vector{ComplexF64},
	fjₒjₛ_r₁_rsrc::Vector{ComplexF64},Grrjₛ_r₂_rsrc::ComplexF64)

	@. Hjₒjₛω_r₁r₂ = conj(fjₒjₛ_r₁_rsrc) * Grrjₛ_r₂_rsrc
end

# Radial components of dG for flows
# Only tangential (+) component
function radial_fn_uniform_rotation_firstborn!(G::AbstractVector{<:Complex},
	Gsrc::AA,Gobs::AA,j,::los_radial) where {AA<:AbstractArray{<:Complex,2}}

	@views @. G = Gsrc[:,0] * Gobs[:,0] -
					Gsrc[:,0] * Gobs[:,1]/Ω(j,0) -
					Gsrc[:,1]/Ω(j,0) * Gobs[:,0] + 
					ζjₒjₛs(j,j,1) * Gsrc[:,1] * Gobs[:,1]
end

function radial_fn_uniform_rotation_firstborn!(G::AbstractMatrix{<:Complex},
	Gsrc::AA,Gobs::AA,j,::los_earth) where {AA<:AbstractArray{<:Complex,3}}

	@views @. G = Gsrc[:,0,0] * Gobs[:,0,:] -
					Gsrc[:,0,0] * Gobs[:,1,:]/Ω(j,0) -
					Gsrc[:,1,0]/Ω(j,0) * Gobs[:,0,:] + 
					ζjₒjₛs(j,j,1) * Gsrc[:,1,0] * Gobs[:,1,:]
end

function Gⱼₒⱼₛω_u⁺_firstborn!(G::AbstractMatrix{<:Complex},
	Gsrc::AA,jₛ::Integer,Gobs::AA,jₒ::Integer,
	::los_radial) where {AA<:AbstractArray{<:Complex,2}}

	# The actual G¹ₗⱼ₁ⱼ₂ω_00(r,rᵢ,rₛ) =  G[:,0] + ζ(jₛ,jₒ,ℓ)G[:,1]
	# We store the terms separately and add them up for each ℓ

	for r_ind in axes(G,1)
		G[r_ind,0] = Gsrc[r_ind,0] * Gobs[r_ind,0] -
							Gsrc[r_ind,0] * Gobs[r_ind,1]/Ω(jₒ,0) -
							Gsrc[r_ind,1]/Ω(jₛ,0) * Gobs[r_ind,0]

		G[r_ind,1] = Gsrc[r_ind,1] * Gobs[r_ind,1]
	end
end

function Gⱼₒⱼₛω_u⁺_firstborn!(G::AbstractArray{<:Complex,3},
	Gsrc::AA,jₛ::Integer,Gobs::AA,jₒ::Integer,
	::los_earth) where {AA<:AbstractArray{<:Complex,3}}

	# The actual G¹ₗⱼ₁ⱼ₂ω_α₁0(r,rᵢ,rₛ) = G[:,0,α₁] + ζ(jₛ,jₒ,ℓ)G[:,1,α₁]
	# We store the terms separately and add them up for each ℓ

	for α₁ in axes(G,2),r_ind in axes(G,1)

		G[r_ind,0,α₁] = Gsrc[r_ind,0,0] * Gobs[r_ind,0,α₁] -
							Gsrc[r_ind,0,0] * Gobs[r_ind,1,α₁]/Ω(jₒ,0) -
							Gsrc[r_ind,1,0]/Ω(jₛ,0) * Gobs[r_ind,0,α₁]

		G[r_ind,1,α₁] = Gsrc[r_ind,1,0] * Gobs[r_ind,1,α₁]
	end
end

function Gⱼₒⱼₛω_u⁰_firstborn!(G::AbstractMatrix{<:Complex},
	drGsrc::AA,jₛ::Integer,Gobs::AA,jₒ::Integer,
	::los_radial) where {AA<:AbstractArray{<:Complex,2}}

	# The actual G⁰ₗⱼ₁ⱼ₂ω_00(r,rᵢ,rₛ) =  G[:,0] + ζ(jₛ,jₒ,ℓ)G[:,1]
	# We store the terms separately and add them up for each ℓ

	@. G = Gobs * drGsrc
end

function Gⱼₒⱼₛω_u⁰_firstborn!(G::AbstractArray{<:Complex,3},
	drGsrc::AA,jₛ::Integer,Gobs::AA,jₒ::Integer,
	::los_earth) where {AA<:AbstractArray{<:Complex,3}}
	
	# The actual G⁰ₗⱼ₁ⱼ₂ω_α₁0(r,rᵢ,rₛ) = G[:,0,α₁] + ζ(jₛ,jₒ,ℓ)G[:,1,α₁]
	# We store the terms separately and add them up for each ℓ
	for α₁ in axes(Gobs,3)
		for r_ind in axes(Gobs,1)
			G[r_ind,0,α₁] = Gobs[r_ind,0,α₁] * drGsrc[r_ind,0,0]
		end
		for r_ind in axes(Gobs,1)
			G[r_ind,1,α₁] = Gobs[r_ind,1,α₁] * drGsrc[r_ind,1,0]
		end
	end
end

# This function computes Gparts
# Components (0) and (+)
function Gⱼₒⱼₛω_u⁰⁺_firstborn!(G::OffsetVector{<:OffsetArray},
	Gsrc::AA,drGsrc::AA,jₛ::Integer,
	Gobs::AA,jₒ::Integer,los::los_direction) where {AA<:AbstractArray{<:Complex}}
	
	Gⱼₒⱼₛω_u⁰_firstborn!(G[0],drGsrc,jₛ,Gobs,jₒ,los)
	Gⱼₒⱼₛω_u⁺_firstborn!(G[1],  Gsrc,jₛ,Gobs,jₒ,los)
end

# This function is used in computing Jₗⱼₒⱼₛω_u⁰⁺_firstborn
# We evaluate Gγₗⱼ₁ⱼ₂ω_00(r,rᵢ,rₛ) = Gsum[r,γ] = G[:,0] + ζ(jₛ,jₒ,ℓ)G[:,1]
# This function is run once for each ℓ using cached values of G
function Gγₗⱼₒⱼₛω_α₁_firstborn!(Gsum::OffsetArray{ComplexF64,2},
	Gparts::OffsetVector{<:OffsetArray{ComplexF64,2}},
	jₒ::Int,jₛ::Int,ℓ::Int)

	coeff = ζjₒjₛs( jₒ,jₛ,ℓ)
	
	G = Gparts[0]
	for r_ind in axes(Gsum,1)
		Gsum[r_ind,0] = G[r_ind,0] + coeff * G[r_ind,1]
	end

	G = Gparts[1]
	for r_ind in axes(Gsum,1)
		Gsum[r_ind,1] = G[r_ind,0] + coeff * G[r_ind,1]
	end
end

# This function is used in computing Jₗⱼₒⱼₛω_u⁰⁺_firstborn
# We evaluate Gγₗⱼ₁ⱼ₂ω_α₁0(r,rᵢ,rₛ) = Gsum[r,γ,α₁] = G[:,0,α₁] + ζ(jₛ,jₒ,ℓ)G[:,1,α₁]
# This function is run once for each ℓ using cached values of G
function Gγₗⱼₒⱼₛω_α₁_firstborn!(Gsum::OffsetArray{ComplexF64,3},
	Gparts::OffsetVector{<:OffsetArray{ComplexF64,3}},
	jₒ::Int,jₛ::Int,ℓ::Int)

	coeff = ζjₒjₛs( jₒ,jₛ,ℓ)
	
	G = Gparts[0]
	for α₁ in axes(Gsum,3), r_ind in axes(Gsum,1)
		Gsum[r_ind,0,α₁] = G[r_ind,0,α₁] + coeff * G[r_ind,1,α₁]
	end

	G = Gparts[1]
	for α₁ in axes(Gsum,3), r_ind in axes(Gsum,1)
		Gsum[r_ind,1,α₁] = G[r_ind,0,α₁] + coeff * G[r_ind,1,α₁]
	end
end

# We compute Hγₗⱼ₁ⱼ₂ω_00(r,r₁,r₂,rₛ) = conj(Gγₗⱼ₁ⱼ₂ω_00(r,r₁,rₛ)) * Gⱼ₂ω_00(r₂,rₛ)
function Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(H::AbstractArray{ComplexF64,2},Gparts,
	Gγₗⱼₒⱼₛω_α₁::AbstractArray{ComplexF64,2},Grr::ComplexF64,jₒ,jₛ,ℓ)

	Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁,Gparts,jₒ,jₛ,ℓ)
	@. H = conj(Gγₗⱼₒⱼₛω_α₁) * Grr
end

# We compute Hγₗⱼ₁ⱼ₂ω_α₁α₂(r,r₁,r₂,rₛ) = conj(Gγₗⱼ₁ⱼ₂ω_α₁0(r,r₁,rₛ)) * Gⱼ₂ω_α₂0(r₂,rₛ)
function Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(H::AbstractArray{ComplexF64,4},Gparts,
	Gγₗⱼₒⱼₛω_α₁_r₁::AbstractArray{ComplexF64,3},G::AbstractVector{ComplexF64},jₒ,jₛ,ℓ)

	Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁_r₁,Gparts,jₒ,jₛ,ℓ)

	@inbounds for α₂ in axes(G,1)
		Gα₂r_r₂ = G[α₂]
		for ind3 in axes(Gγₗⱼₒⱼₛω_α₁_r₁,3), 
			ind2 in axes(Gγₗⱼₒⱼₛω_α₁_r₁,2),
			ind1 in axes(Gγₗⱼₒⱼₛω_α₁_r₁,1)
				
			H[ind1,ind2,ind3,α₂] = conj(Gγₗⱼₒⱼₛω_α₁_r₁[ind1,ind2,ind3]) * Gα₂r_r₂
		end
	end
end

end # module

################################################################
# Three dimensional Green function
################################################################

module Greenfn_3D

using ..Continuous_FFT
using ..Greenfn_radial
using ..Timer_utils

using DistributedArrays
using DSP
using FastGaussQuadrature
using LegendrePolynomials
using NumericallyIntegrateArrays
using PointsOnASphere
using TwoPointFunctions

import ParallelUtilities: finalize_except_wherewhence

import ..Greenfn_radial: Gfn_path_from_source_radius, radial_grid_index

export Gfn_path_from_source_radius
export get_θϕ
export Grrω
export Grrt
export ξrω
export ξrt
export δGrr_uniform_rotation_firstborn_integrated_over_angle
export δGrr_uniform_rotation_rotatedwaves_linearapprox
export δGrr_uniform_rotation_rotatedwaves
export δGrr_uniform_rotation_firstborn_integrated_over_angle_numerical
export δGrr_soundspeed_firstborn_integrated_over_angle
export δGrr_soundspeed_full

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

#######################################################################################################

import ..HelioseismicKernels: Powspec

function ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)
	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ν_ind_range = get(kwargs,:ν_ind_range,axes(ν_arr,1))
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)
	np = ParallelUtilities.nworkersactive(modes_iter.iterators)
	ℓ_range,ν_ind_range,modes_iter,np
end

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

#######################################################################################################

# Gfn_path_from_source_radius(x::Point3D;kwargs...) = Gfn_path_from_source_radius(x.r;kwargs...)

function int_over_r_and_contract(d::AbstractArray{T,N}) where {T,N}
	d_local = localpart(d)
	T_int = promote_type(Float64,T)
	int_d = zeros(T_int,axes(d_local)[2:end]...)
	inds_trailing = CartesianIndices(axes(d_local)[2:end])
	for ind in inds_trailing
		int_d[ind] = simps(r.^2 .* d_local[:,ind],r)
	end
	int_d = dropdims(sum(int_d,dims=1),dims=1)
	return int_d
end

function int_over_r(d)
	d_local = localpart(d)
	T = eltype(d)
	T_int = promote_type(Float64,T)
	int_d = zeros(T_int,axes(d_local)[2:end]...)
	inds_trailing = CartesianIndices(axes(d_local)[2:end])
	for ind in inds_trailing
		int_d[ind] = simps(r.^2 .* d_local[:,ind],r)
	end
	return int_d
end

function permutedims_rvΩ_to_ϕθvr(d::DArray,nϕ)
	nr,nv,nΩ = size(d)
	nθ = div(nΩ,nϕ)
	# Get full (r,v,Ω) array
	arr = @spawnat procs(d)[1] Array(d)
	# Convert to (Ω,v,r)
	arr = @spawnat procs(d)[1] permutedims(fetch(arr),[3,2,1])
	# Convert to (θ,ϕ,v,r)
	arr = @spawnat procs(d)[1] reshape(fetch(arr),nθ,nϕ,nv,nr)
	# Convert to (ϕ,θ,v,r)
	arr = @spawnat procs(d)[1] permutedims(fetch(arr),[2,1,3,4])
	# Distribute over r
	@fetchfrom procs(d)[1] distribute(fetch(arr),procs=workers(),dist=[1,1,1,nworkers()])
end

function permutedims_ϕθvr_to_rvΩ(d::DArray)
	nϕ,nθ,nv,nr = size(d)
	nΩ = nθ*nϕ
	# Get full (ϕ,θ,v,r) array
	arr = @spawnat procs(d)[1] Array(d)
	# Convert to (θ,ϕ,v,r)
	arr = @spawnat procs(d)[1] permutedims(fetch(arr),[2,1,3,4])
	# Convert to (Ω,v,r)
	arr = @spawnat procs(d)[1] reshape(fetch(arr),nΩ,nv,nr)
	# Convert to (r,v,Ω)
	arr = @spawnat procs(d)[1] permutedims(fetch(arr),[3,2,1])
	# Distribute over Ω
	np_Ω = min(nworkers(),nΩ)
	@fetchfrom procs(d)[1] distribute(fetch(arr),procs=workers()[1:np_Ω],dist=[1,1,np_Ω])
end

function get_θϕ(ℓmax;kwargs...)
	if !isnothing(get(kwargs,:θ,nothing))
		θ_full = kwargs[:θ]
		_,weights = gausslegendre(length(θ_full))
	else
		nodes,weights = gausslegendre(ℓmax+1)
		θ_full = acos.(nodes)
		p = sortperm(θ_full)
		permute!(θ_full,p)
		permute!(nodes,p)
		permute!(weights,p)
	end

	nϕ = get(kwargs,:nϕ,4ℓmax+2)
	ϕ_full = get(kwargs,:ϕ,LinRange(0,2π,nϕ+1)[1:end-1])
	θ_ϕ_iterator = Iterators.product(θ_full,ϕ_full)
	nΩ = length(θ_ϕ_iterator)
	return θ_full,ϕ_full,θ_ϕ_iterator,nΩ,weights
end

function G3D_helicity(x′::Point3D,ν::Real=3e-3;kwargs...)

	c_scale = get(kwargs,:c_scale,1)
	Gfn_path_src = Gfn_path_from_source_radius(x′,c_scale=c_scale)

	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν_Gfn
	
	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = ℓ_range[end]

	θ_full,ϕ_full,θ_ϕ_iterator,nΩ = get_θϕ(ℓmax;kwargs...)

	ν_test_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_range = ν_test_ind:ν_test_ind
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)
	proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,
			1:Nν_Gfn,modes_iter,num_procs)

	G_components = get(kwargs,:G_components,-1:1)

	np = min(nworkers(),nΩ)

	Gfn_3D_darr = DArray((nr,length(G_components),nΩ),workers()[1:np],[1,1,np]) do inds

		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)
		Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1)
		G00 = view(Gsrc,:,0)
		G10 = view(Gsrc,:,1)

		Gfn_3D_local = zeros(ComplexF64,inds[1],G_components,inds[3])
		θ_ϕ_inds_local = last(inds)
		θ_ϕ_iterator_local = Iterators.take(
			Iterators.drop(θ_ϕ_iterator,first(θ_ϕ_inds_local)-1),
			length(θ_ϕ_inds_local))

		d01Pl_cosχ = OffsetArray{Float64}(undef,0:ℓmax,0:1)
		Pl_cosχ = view(d01Pl_cosχ,:,0)
		dPl_cosχ = view(d01Pl_cosχ,:,1)

		for (ℓ,ω_ind) in modes_iter

			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,:,1:2,1,1)

			for (θϕ_ind,(θ,ϕ)) in zip(θ_ϕ_inds_local,θ_ϕ_iterator_local)
			
				Pl_dPl!( d01Pl_cosχ, cosχ((θ,ϕ),x′) )
			
				# (-1,0) component
				if -1 ∈ G_components
					em1_dot_∇_n_dot_n′ = 1/√2 * (∂θ₁cosχ((θ,ϕ),x′) + im * ∇ϕ₁cosχ((θ,ϕ),x′) )
					@. Gfn_3D_local[:,-1,θϕ_ind] += (2ℓ +1)/4π  * G10/Ω(ℓ,0) * 
												dPl_cosχ[ℓ] * em1_dot_∇_n_dot_n′
				end

				if 0 ∈ G_components
					# (0,0) component
					@. Gfn_3D_local[:,0,θϕ_ind] +=  (2ℓ +1)/4π * G00 * Pl_cosχ[ℓ]
				end
				
				# (1,0) component
				if 1 ∈ G_components
					e1_dot_∇_n_dot_n′ = 1/√2 * (-∂θ₁cosχ((θ,ϕ),x′) + im * ∇ϕ₁cosχ((θ,ϕ),x′) )
					@. Gfn_3D_local[:,1,θϕ_ind] +=  (2ℓ +1)/4π  * G10/Ω(ℓ,0) * 
												dPl_cosχ[ℓ] * e1_dot_∇_n_dot_n′
				end

			end
		end

		close.(values(Gfn_fits_files_src))
	end

	# @sync for p in procs(Gfn_3D_darr)
	#   @async @spawnat p compute_angular_part()
	# end

	# Gfn_3D_arr = Array(Gfn_3D_darr)
	# close(Gfn_3D_darr)
	# Gfn_3D_arr = reshape(Gfn_3D_arr,nr,nθ,nϕ,3)
	# return Gfn_3D_darr
end

function G3D_spherical(x′::Point3D,ν::Real=3e-3;kwargs...)
	
	c_scale = get(kwargs,:c_scale,1)
	Gfn_path_src = Gfn_path_from_source_radius(x′,c_scale=c_scale)

	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν_Gfn
	
	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = ℓ_range[end]

	θ_full,ϕ_full,θ_ϕ_iterator,nΩ = get_θϕ(ℓmax;kwargs...)

	ν_test_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_range = ν_test_ind:ν_test_ind
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)
	# Get procids for all ℓs and one ν
	proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,
		1:Nν_Gfn,modes_iter,num_procs)

	G_components = get(kwargs,:G_components,1:3)

	np = min(nworkers(),nΩ)

	Gfn_3D_darr = DArray((nr,length(G_components),nΩ),workers()[1:np],[1,1,np]) do inds
		
		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

		Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1)
		G00 = view(Gsrc,:,0)
		G10 = view(Gsrc,:,1)

		Gfn_3D_local = zeros(ComplexF64,inds...)
		θ_ϕ_inds_local = last(inds)
		θ_ϕ_iterator_local = Iterators.take(
			Iterators.drop(θ_ϕ_iterator,first(θ_ϕ_inds_local)-1),
			length(θ_ϕ_inds_local))

		d01Pl_cosχ = OffsetArray{Float64}(undef,0:ℓmax,0:1)
		Pl_cosχ = view(d01Pl_cosχ,:,0)
		dPl_cosχ = view(d01Pl_cosχ,:,1)

		for (ℓ,ω_ind) in modes_iter

			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,:,1:2,1,1)

			for (θϕ_ind,(θ,ϕ)) in zip(θ_ϕ_inds_local,θ_ϕ_iterator_local)
			
				Pl_dPl!( d01Pl_cosχ, cosχ((θ,ϕ),x′) )

				# (r,r) component
				if 1 ∈ G_components
					@. Gfn_3D_local[:,1,θϕ_ind] +=  (2ℓ+1)/4π * G00 * Pl_cosχ[ℓ]
				end

				# (θ,r) component
				if 2 ∈ G_components
					eθ_dot_∇_n_dot_n′ = ∂θ₁cosχ((θ,ϕ),x′)
					∇θ_Pl = dPl_cosχ[ℓ] * eθ_dot_∇_n_dot_n′
					@. Gfn_3D_local[:,2,θϕ_ind] += (2ℓ+1)/4π * G10/Ω(ℓ,0) * ∇θ_Pl
				end
				
				# (ϕ,r) component
				if 3 ∈ G_components
					eϕ_dot_∇_n_dot_n′ = ∇ϕ₁cosχ((θ,ϕ),x′)
					∇ϕ_Pl = dPl_cosχ[ℓ] * eϕ_dot_∇_n_dot_n′
					@. Gfn_3D_local[:,3,θϕ_ind] +=  (2ℓ+1)/4π * G10/Ω(ℓ,0) * ∇ϕ_Pl
				end
			end
		end

		close.(values(Gfn_fits_files_src))

		Gfn_3D_local.parent
	end
end

function divGr3D(xsrc::Point3D,ν::Real=3e-3;kwargs...)
	
	c_scale = get(kwargs,:c_scale,1)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc,c_scale=c_scale)

	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν_Gfn
	
	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = ℓ_range[end]

	θ_full,ϕ_full,θ_ϕ_iterator,nΩ = get_θϕ(ℓmax;kwargs...)

	ν_test_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_range = ν_test_ind:ν_test_ind
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)
	# Get procids for all ℓs and one ν
	proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,
		1:Nν_Gfn,modes_iter,num_procs)

	G_components = get(kwargs,:G_components,1:3)

	np = min(nworkers(),nΩ)

	Gfn_3D_darr = DArray((nr,nΩ),workers()[1:np],[1,np]) do inds
		
		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

		Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1)
		drGsrc = zeros_Float64_to_ComplexF64(1:nr,0:0)
		divGr_src_ℓ = zeros(ComplexF64,nr)

		divG = zeros(ComplexF64,inds...)
		θ_ϕ_inds_local = last(inds)
		θ_ϕ_iterator_local = Iterators.take(
			Iterators.drop(θ_ϕ_iterator,first(θ_ϕ_inds_local)-1),
			length(θ_ϕ_inds_local))

		angles_with_index = zip(θ_ϕ_inds_local,θ_ϕ_iterator_local)

		Pl_cosχ = zeros(0:ℓmax)

		for (ℓ,ω_ind) in modes_iter

			# Green function about the source location
			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,:,1:2,1,1)

			# Derivative of the Green function about the source location
			read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,:,1:1,1,2)

			divGr_src_ℓ .= divG_radial(ℓ,Gsrc,drGsrc)

			for (θϕ_ind,θϕ) in angles_with_index
			
				Pl!(Pl_cosχ,cosχ(θϕ,xsrc),lmax=ℓmax)

				@. divG[:,θϕ_ind] += (2ℓ+1)/4π * divGr_src_ℓ * Pl_cosχ[ℓ]
				
			end
		end

		close.(values(Gfn_fits_files_src))

		divG.parent
	end
end

function divGr2D_robs(xsrc::Point3D,ν::Real=3e-3;kwargs...)
	
	c_scale = get(kwargs,:c_scale,1)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc,c_scale=c_scale)

	r_obs = get(kwargs,:r_obs,r_obs_default)
	r_obs_ind = radial_grid_index(r_obs)

	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν_Gfn
	
	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = ℓ_range[end]

	θ_full,ϕ_full,θ_ϕ_iterator,nΩ = get_θϕ(ℓmax;kwargs...)

	ν_test_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_range = ν_test_ind:ν_test_ind
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)
	# Get procids for all ℓs and one ν
	proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,
		1:Nν_Gfn,modes_iter,num_procs)

	G_components = get(kwargs,:G_components,1:3)

	np = min(nworkers(),nΩ)

	Gfn_3D_darr = DArray((1,nΩ),workers()[1:np],[1,np]) do inds
		
		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

		Gsrc = zeros_Float64_to_ComplexF64(0:1)
		drGsrc = zeros_Float64_to_ComplexF64(0:0)
		divGr_src_ℓ = zeros(ComplexF64,r_obs_ind:r_obs_ind)

		divG = zeros(ComplexF64,inds...)
		θ_ϕ_inds_local = last(inds)
		θ_ϕ_iterator_local = Iterators.take(
			Iterators.drop(θ_ϕ_iterator,first(θ_ϕ_inds_local)-1),
			length(θ_ϕ_inds_local))

		angles_with_index = zip(θ_ϕ_inds_local,θ_ϕ_iterator_local)

		Pl_cosχ = zeros(0:ℓmax)

		for (ℓ,ω_ind) in modes_iter

			# Green function about the source location
			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,r_obs_ind,1:2,1,1)

			# Derivative of the Green function about the source location
			read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,r_obs_ind,1:1,1,2)

			divGr_src_ℓ .= divG_radial(ℓ,Gsrc,drGsrc)

			for (θϕ_ind,θϕ) in angles_with_index
			
				Pl!(Pl_cosχ,cosχ(θϕ,xsrc),lmax=ℓmax)

				@. divG[:,θϕ_ind] += (2ℓ+1)/4π * divGr_src_ℓ * Pl_cosχ[ℓ]
				
			end
		end

		close.(values(Gfn_fits_files_src))

		divG.parent
	end
end

function Grrω_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs::Point3D,xsrc::Point3D,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	c_scale = 1,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=xsrc.r,c_scale=c_scale)

	Gfn_path_src,NGfn_files = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,Nν_Gfn = p_Gsrc

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators
	ℓmax = maximum(ℓ_range)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,(ℓ_arr,1:Nν_Gfn),
		ℓ_ωind_iter_on_proc,NGfn_files)

	Gω_proc = zeros(ComplexF64,ν_ind_range)

	α_robs = zeros_Float64_to_ComplexF64()

	Pl_cosχ = Pl(cosχ(xsrc,xobs),lmax=ℓmax)

	r_obs_ind = radial_grid_index(xobs)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		read_Gfn_file_at_index!(α_robs,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files,r_obs_ind,1,1,1)

		Gω_proc[ω_ind] += (2ℓ+1)/4π * α_robs[] * Pl_cosχ[ℓ]

		signaltomaster!(progress_channel)
	end

	closeGfnfits(Gfn_fits_files_src)

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))
	
	parent(Gω_proc)
end

function Grrω(xobs::Point3D,xsrc::Point3D;kwargs...)

	c_scale = get(kwargs,:c_scale,1)
	p_Gsrc = read_all_parameters(r_src=xsrc.r,c_scale=c_scale)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	Gω_in_range = pmapsum_timed(Grrω_partial,modes_iter,
		xobs,xsrc,p_Gsrc,c_scale;
		progress_str="Modes summed in Grrω : ")

	Grrω = pad_zeros_ν(Gω_in_range,ν_ind_range,
			Nν_Gfn,ν_start_zeros,ν_end_zeros)

	@save_to_fits_and_return(Grrω)
end

function Grrω(xobs::Point3D,xsrc::Point3D,ν::Real;kwargs...)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc,c_scale=get(kwargs,:c_scale,1))
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_start_zeros ν_arr

	ν_ind = searchsortedfirst(ν_arr,ν)

	Grrω(xobs,xsrc;kwargs...,ν_ind_range=ν_ind:ν_ind)[ν_start_zeros + ν_ind]
end

function ∂ϕobsGrrω(xobs::Point3D,xsrc::Point3D;kwargs...)
	c_scale = get(kwargs,:c_scale,1)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc,c_scale=c_scale)

	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν_Gfn ν_start_zeros dω Nν
	
	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = maximum(ℓ_range)

	ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)

	function summodes(ℓ_ωind_iter_on_proc)

		proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

		Gω_proc = zeros(ComplexF64,ν_ind_range .+ ν_start_zeros)

		α_robs = zeros_Float64_to_ComplexF64()

		∂ϕobsPl_cosχ = dPl(cosχ(xsrc,xobs),lmax=ℓmax).*∂ϕ₁cosχ(xobs,xsrc)

		r_obs_ind = radial_grid_index(xobs)

		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

			proc_id_mode_Gsrc,ℓω_index_in_file = 
				procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs)
			Gsrc_file = Gfn_fits_files_src[proc_id_mode_Gsrc]

			read_Gfn_file_at_index!(α_robs,Gsrc_file,
				ℓω_index_in_file,r_obs_ind,1,1,1)

			Gω_proc[ω_ind] += (2ℓ+1)/4π * α_robs[] * ∂ϕobsPl_cosχ[ℓ]
		end

		close.(values(Gfn_fits_files_src))
		
		parent(Gω_proc)
	end

	Gω_in_range = pmapsum(summodes,modes_iter)

	Grrω = pad_zeros_ν(Gω_in_range,ν_ind_range,
			Nν_Gfn,ν_start_zeros,ν_end_zeros)
end

function ξrω(xobs::Point3D,xsrc::Point3D;kwargs...)
	G = Grrω(xobs,xsrc;kwargs...)
	
	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_full

	ω = 2π.*ν_full
	P_ω = Powspec.(ω)

	G.*P_ω
end

function Grrt(xobs::Point3D,xsrc::Point3D;kwargs...)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc,c_scale=get(kwargs,:c_scale,1))

	@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt
	τ_ind_range = get(kwargs,:τ_ind_range,1:Nt)
	Gω = Grrω(xobs,xsrc;kwargs...)
	OffsetArray(fft_ω_to_t(Gω,dν)[τ_ind_range],τ_ind_range)
end

function ξrt(xobs::Point3D,xsrc::Point3D;kwargs...)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_path_src,"parameters.jld2") dν
	ξ = ξrω(xobs,xsrc;kwargs...)
	fft_ω_to_t(ξ,dν)
end

function ∂ϕobsGrrt(xobs::Point3D,xsrc::Point3D;kwargs...)
	c_scale = get(kwargs,:c_scale,1)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc,c_scale=c_scale)

	@load joinpath(Gfn_path_src,"parameters.jld2") dν
	f = ∂ϕobsGrrω(xobs,xsrc;kwargs...)
	fft_ω_to_t(f,dν)
end

function Grrt_rotating(xobs::Point3D,xsrc::Point3D;kwargs...)

	# Return Grr(x2,x1,ω) = RFFT(G(x2,x1,t)) = RFFT(IRFFT(G0(x2 - Ωτ,x1,ω))(t))

	Gfn_path_src = Gfn_path_from_source_radius(xsrc,c_scale=get(kwargs,:c_scale,1))
	@load(joinpath(Gfn_path_src,"parameters.jld2"),
		Nt,dt,Nν,dν,ℓ_arr,Nν_Gfn,num_procs,ν_start_zeros,ν_end_zeros)

	τ_ind_range = get(kwargs,:τ_ind_range,1:Nt)
	Nτ = length(τ_ind_range)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = maximum(ℓ_range)

	ν_ind_range = 1:Nν_Gfn
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)

	# The first step is loading in the αωℓ

	function summodes(ℓ_ωind_iter_on_proc)

		proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,
							ℓ_ωind_iter_on_proc,num_procs)
		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

		αℓω = zeros(ComplexF64,ℓ_range,Nν_Gfn)
		G = zeros_Float64_to_ComplexF64()

		r_obs_ind = radial_grid_index(xobs)

		# Read all radial parts
		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

			read_Gfn_file_at_index!(G,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,r_obs_ind,1,1,1)

			αℓω[ℓ,ω_ind] = G[]
		end

		close.(values(Gfn_fits_files_src))
		copy(transpose(αℓω))
	end

	αωℓ = pmapsum(summodes,modes_iter)
	αωℓ = vcat(zeros(ν_start_zeros,ℓ_range),αωℓ,
			zeros(ν_end_zeros,ℓ_range))
	αtℓ = fft_ω_to_t(αωℓ,dν)
	αℓt = permutedims(αtℓ)[:,τ_ind_range]
	np = min(nworkers(),Nτ)
	αℓt = distribute(αℓt,procs=workers()[1:np],dist=[1,np])                                  

	Gt = DArray((Nτ,),workers()[1:np],[np]) do inds
		τ_ind_range_proc = τ_ind_range[first(inds)]
		αℓt_local = OffsetArray(αℓt[:lp],ℓ_range,τ_ind_range_proc)
		Gt = zeros(τ_ind_range_proc)
		for τ_ind in τ_ind_range_proc
			# τ goes from -T/2 to T/2-dt but is fftshifted
			τ = (τ_ind<=div(Nt,2) ? (τ_ind-1) : (τ_ind-1- Nt)) * dt
			xobs′ = Point3D(xobs.r,xobs.θ,xobs.ϕ-Ω_rot*τ)
			# xsrc′ = Point3D(xsrc.r,xsrc.θ,xsrc.ϕ-Ω_rot*τ)
			Pl_cosχ = Pl(cosχ(xsrc,xobs′),lmax=ℓmax)
			for ℓ in ℓ_range
				Gt[τ_ind] += (2ℓ+1)/4π * αℓt_local[ℓ,τ_ind] * Pl_cosχ[ℓ]
			end
		end
		Gt.parent
	end
	
	OffsetArray(Array(Gt),τ_ind_range)
end

function Grrt_rotating_nonrotating(xobs::Point3D,xsrc::Point3D;kwargs...)

	# Return Grr(x2,x1,ω) = RFFT(G(x2,x1,t)) = RFFT(IRFFT(G0(x2 - Ωτ,x1,ω))(t))

	Gfn_path_src = Gfn_path_from_source_radius(xsrc,c_scale=get(kwargs,:c_scale,1))
	@load(joinpath(Gfn_path_src,"parameters.jld2"),
		Nt,dt,Nν,dν,ℓ_arr,Nν_Gfn,num_procs,ν_start_zeros,ν_end_zeros)

	τ_ind_range = get(kwargs,:τ_ind_range,1:Nt)
	Nτ = length(τ_ind_range)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = maximum(ℓ_range)

	ν_ind_range = 1:Nν_Gfn
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)

	# The first step is loading in the αωℓ

	function summodes(ℓ_ωind_iter_on_proc)

		proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,
							ℓ_ωind_iter_on_proc,num_procs)
		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

		αℓω = zeros(ComplexF64,ℓ_range,Nν_Gfn)
		G = zeros_Float64_to_ComplexF64()

		r_obs_ind = radial_grid_index(xobs)

		# Read all radial parts
		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

			read_Gfn_file_at_index!(G,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,r_obs_ind,1,1,1)

			αℓω[ℓ,ω_ind] = G[]
		end

		close.(values(Gfn_fits_files_src))
		copy(transpose(αℓω))
	end

	αωℓ = pmapsum(summodes,modes_iter)
	αωℓ = vcat(zeros(ν_start_zeros,ℓ_range),αωℓ,
		zeros(ν_end_zeros,ℓ_range))
	αtℓ = fft_ω_to_t(αωℓ,dν)
	αℓt = permutedims(αtℓ)[:,τ_ind_range]
	np = min(nworkers(),Nτ)
	αℓt = distribute(αℓt,procs=workers()[1:np],dist=[1,np])                                  

	Gt = DArray((2,Nτ),workers()[1:np],[1,np]) do inds
		τ_ind_range_proc = τ_ind_range[last(inds)]
		αℓt_local = OffsetArray(αℓt[:lp],ℓ_range,τ_ind_range_proc)
		Gt = zeros(2,τ_ind_range_proc)
		Pl_cosχ = Pl(cosχ(xsrc,xobs),lmax=ℓmax)
		for τ_ind in τ_ind_range_proc
			# τ goes from -T/2 to T/2-dt but is fftshifted
			τ = (τ_ind<=div(Nt,2) ? (τ_ind-1) : (τ_ind-1- Nt)) * dt
			xobs′ = Point3D(xobs.r,xobs.θ,xobs.ϕ-Ω_rot*τ)
			# xsrc′ = Point3D(xsrc.r,xsrc.θ,xsrc.ϕ-Ω_rot*τ)
			Pl_cosχ′ = Pl(cosχ(xsrc,xobs′),lmax=ℓmax)
			for ℓ in ℓ_range
				Gt[1,τ_ind] += (2ℓ+1)/4π * αℓt_local[ℓ,τ_ind] * Pl_cosχ′[ℓ]
				Gt[2,τ_ind] += (2ℓ+1)/4π * αℓt_local[ℓ,τ_ind] * Pl_cosχ[ℓ]
			end
		end
		Gt.parent
	end
	
	OffsetArray(permutedims(Array(Gt)),τ_ind_range,1:2)
end

function u_dot_∇G_uniform_rotation(x′,ν=3e-3;kwargs...)
	# We compute this assuming u=Ωrsin(θ)eϕ. This means u.∇ = Ω∂ϕ.
	# Therefore we need to evaluate Ω∂ϕG
	# G = Grr er + Gθr eθ + Gϕr eϕ
	#   = G00 Pl(cosχ) er + G10/Ω(ℓ,0)((eθ⋅∇Pl(cosχ)) eθ  + (eϕ⋅∇Pl(cosχ)) eϕ)
	# ∂ϕG =  ∂ϕGrr er + ∂ϕGθr eθ + ∂ϕGϕr eϕ + Grr ∂ϕer + Gθr ∂ϕeθ + Gϕr ∂ϕeϕ
	# We use ∂ϕer = sinθ eϕ, ∂ϕeθ = cosθ eϕ, ∂ϕeϕ = -cosθ eθ - sinθ er
	# Substituting we obtain 
	# ∂ϕG = ∂ϕGrr er + ∂ϕGθr eθ + ∂ϕGϕr eϕ + Grr sinθ eϕ + Gθr cosθ eϕ + Gϕr (-cosθ eθ - sinθ er)
	#     = (∂ϕGrr - Gϕr sinθ) er + (∂ϕGθr - Gϕr cosθ) eθ + (∂ϕGϕr + Grr sinθ + Gθr cosθ) eϕ
	#     = (G00 - G10/Ω(ℓ,0) sinθ)∂ϕPl er + G10/Ω(ℓ,0)(∂ϕ∂θPl - cosθ/sinθ ∂ϕPl) eθ 
	#       + (G10/Ω(ℓ,0)(1/sinθ ∂²ϕPl + cosθ ∂θPl ) + G00 Pl sinθ ) eϕ

	Gfn_path_src = Gfn_path_from_source_radius(x′)

	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν_Gfn
	
	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = maximum(ℓ_range)

	θ_full,ϕ_full,θ_ϕ_iterator,nΩ = get_θϕ(ℓmax;kwargs...)
	nθ = size(θ_full,1)
	nϕ = size(ϕ_full,1)

	ν_test_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_range = ν_test_ind:ν_test_ind
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)
	proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,
			1:Nν_Gfn,modes_iter,num_procs)

	np = min(nworkers(),nΩ)

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)

	uϕ∇ϕG_darr = DArray((nr,3,nΩ),workers()[1:np],[1,1,np]) do inds

		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)
		Gsrc = zeros(ComplexF64,nr,0:1);
		G00 = view(Gsrc,:,0);
		G10 = view(Gsrc,:,1);

		uϕ∇ϕG_local = zeros(ComplexF64,inds...);
		θ_ϕ_inds_local = last(inds)
		θ_ϕ_iterator_local = Iterators.take(
			Iterators.drop(θ_ϕ_iterator,first(θ_ϕ_inds_local)-1),
			length(θ_ϕ_inds_local))

		d02Pl_cosχ = OffsetArray{Float64}(undef,0:ℓmax,0:2);
		Pl_cosχ = view(d02Pl_cosχ,:,0);
		dPl_cosχ = view(d02Pl_cosχ,:,1);
		d2Pl_cosχ = view(d02Pl_cosχ,:,2);

		for (ℓ,ω_ind) in modes_iter

			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,:,1:2,1,1)

			for (θϕ_ind,(θ,ϕ)) in zip(θ_ϕ_inds_local,θ_ϕ_iterator_local)
			
				Pl_dPl_d2Pl!( d02Pl_cosχ, cosχ((θ,ϕ),x′) )

				# (r,r) component
				∂ϕPl = dPl_cosχ[ℓ] * ∂ϕ₁cosχ((θ,ϕ),x′)
				@. uϕ∇ϕG_local[:,1,θϕ_ind] +=  (2ℓ+1)/4π * (G00-G10/Ω(ℓ,0)) * ∂ϕPl
				
				# (θ,r) component
				d2Pl_∂θcosχ_∂ϕcosχ = d2Pl_cosχ[ℓ] * ∂ϕ₁cosχ((θ,ϕ),x′) * ∂θ₁cosχ((θ,ϕ),x′)
				@. uϕ∇ϕG_local[:,2,θϕ_ind] += (2ℓ+1)/4π  * G10/Ω(ℓ,0) * d2Pl_∂θcosχ_∂ϕcosχ

				# (ϕ,r) component
				dϕ∇ϕPl = d2Pl_cosχ[ℓ] * ∇ϕ₁cosχ((θ,ϕ),x′) * ∂ϕ₁cosχ((θ,ϕ),x′) + dPl_cosχ[ℓ] * ∇ϕ₁∂ϕ₁cosχ((θ,ϕ),x′)
				∂θPl =  dPl_cosχ[ℓ] * ∂θ₁cosχ((θ,ϕ),x′)
				@. uϕ∇ϕG_local[:,3,θϕ_ind] +=  (2ℓ+1)/4π  * (G10/Ω(ℓ,0) * (dϕ∇ϕPl + cos(θ) * ∂θPl ) + G00 * sin(θ) * Pl_cosχ[ℓ])

			end
		end

		close.(values(Gfn_fits_files_src))

		Ω_rot .* uϕ∇ϕG_local.parent
	end
end

function u_dot_∇G_uniform_rotation_fft_deriv(x,ν=3e-3;kwargs...)
	# We compute this assuming u=Ωrsin(θ)eϕ. This means u.∇ = Ω∂ϕ.
	# Therefore we need to evaluate Ω∂ϕG
	# G = Grr er + Gθr eθ + Gϕr eϕ
	# ∂ϕG =  ∂ϕGrr er + ∂ϕGθr eθ + ∂ϕGϕr eϕ + Grr ∂ϕer + Gθr ∂ϕeθ + Gϕr ∂ϕeϕ
	# We use ∂ϕer = sinθ eϕ, ∂ϕeθ = cosθ eϕ, ∂ϕeϕ = -cosθ eθ - sinθ er
	# Substituting we obtain 
	# ∂ϕG = ∂ϕGrr er + ∂ϕGθr eθ + ∂ϕGϕr eϕ + Grr sinθ eϕ + Gθr cosθ eϕ + Gϕr (-cosθ eθ - sinθ er)
	#     = (∂ϕGrr - Gϕr sinθ) er + (∂ϕGθr - Gϕr cosθ) eθ + (∂ϕGϕr + Grr sinθ + Gθr cosθ) eϕ

	Gfn_path_src = Gfn_path_from_source_radius(x)
	@load joinpath(Gfn_path_src,"parameters.jld2") ℓ_arr

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = ℓ_range[end]

	θ_full,ϕ_full,θ_ϕ_iterator,nΩ = get_θϕ(ℓmax;kwargs...)
	nθ = length(θ_full)
	nϕ = length(ϕ_full);

	G_d = G3D_spherical(x,ν;θ=θ_full,ϕ=ϕ_full,kwargs...)
	np_Ω = length(procs(G_d))

	G_d_ϕθvr = permutedims_rvΩ_to_ϕθvr(G_d,nϕ)
	close(G_d)

	# Compute the ϕ derivative using FFTs as the G components are periodic in ϕ 
	# Fourier modes along ϕ are denoted by m
	# assuming an even number of ϕs
	# Set Nyquist to zero to evaluate derivatives correctly
	m = vcat(0:div(nϕ,2)-1,0,-div(nϕ,2)+1:-1)

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)

	u∂ϕG = DArray((nϕ,nθ,3,nr),workers(),[1,1,1,nworkers()]) do inds

		G = G_d_ϕθvr[:lp]

		# Derivative of the component fields
		∂ϕG = ifft(fft(G,1) .* (im.*m),1)

		# Derivative of the unit vectors
		Gr = view(G,:,:,1,:)
		Gθ = view(G,:,:,2,:)
		Gϕ = view(G,:,:,3,:)

		for r_ind in last(axes(∂ϕG)),(θ_ind,θ) in enumerate(θ_full),
			(ϕ_ind,ϕ) in enumerate(ϕ_full)
			
			∂ϕG[ϕ_ind,θ_ind,1,r_ind] -= Gϕ[ϕ_ind,θ_ind,r_ind]*sin(θ)
			∂ϕG[ϕ_ind,θ_ind,2,r_ind] -= Gϕ[ϕ_ind,θ_ind,r_ind]*cos(θ)
			∂ϕG[ϕ_ind,θ_ind,3,r_ind] += Gr[ϕ_ind,θ_ind,r_ind]*sin(θ) + 
										Gθ[ϕ_ind,θ_ind,r_ind]*cos(θ)
		end
		Ω_rot.*∂ϕG
	end

	close(G_d_ϕθvr)

	d = permutedims_ϕθvr_to_rvΩ(u∂ϕG)
	close(u∂ϕG)
	return d
end

function δLG_uniform_rotation(xsrc,ν=3e-3;kwargs...)
	
	Gfn_save_directory = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr

	ν = ν_arr[searchsortedfirst(ν_arr,ν)]; ω = 2π*ν

	udot∇G = u_dot_∇G_uniform_rotation(xsrc,ν;kwargs...)
	@. udot∇G *= 2im*ω*ρ
end

function δLG_uniform_rotation_fft_deriv(xsrc,ν=3e-3;kwargs...)
	# This is simply 2iωρ u⋅∇ G
	
	Gfn_save_directory = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_save_directory,"parameters.jld2") ν_arr

	ν = ν_arr[searchsortedfirst(ν_arr,ν)]; ω = 2π*ν

	udot∇G = u_dot_∇G_uniform_rotation_fft_deriv(xsrc,ν;kwargs...)
	@. udot∇G *= 2im*ω*ρ
end

function δGrr_uniform_rotation_firstborn_integrated_over_angle_numerical(xobs,xsrc,ν=3e-3;kwargs...)
	# δG_ik(xobs,xsrc) = -∫dx Gij(x1,x)[δL(x)G(x,x2)]jk = -∫dx Gji(x,x1)[δL(x)G(x,x2)]jk
	# We compute δG_rr(xobs,xsrc) = -∫dx Gjr(x,x1)[δL(x)G(x,x2)]jr

	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs

	ν_on_grid = ν_arr[searchsortedfirst(ν_arr,ν)]

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = ℓ_range[end]
	
	θ_full,ϕ_full,_,_,wGL = get_θϕ(ℓmax;kwargs...)
	nθ = length(θ_full)
	dϕ = ϕ_full[2] - ϕ_full[1]; nϕ = length(ϕ_full)

	GδLG = δLG_uniform_rotation(xsrc,ν_on_grid;θ=θ_full,ϕ=ϕ_full,kwargs...)
	G = G3D_spherical(xobs,ν_on_grid;θ=θ_full,ϕ=ϕ_full,kwargs...)
	GδLG .*= G
	close(G)

	integrand_θϕ_futures = [@spawnat p int_over_r_and_contract(GδLG) for p in procs(GδLG)]
	integrand_θϕ = reshape(vcat(fetch.(integrand_θϕ_futures)...),nθ,nϕ)
	close(GδLG)
	dG = - sum(wGL.*integrand_θϕ)*dϕ
end

function δGrr_uniform_rotation_firstborn_integrated_over_angle(xobs,xsrc,ν=3e-3;kwargs...)
	# δG_ik(xobs,xsrc) = -∫dx Gij(x1,x)[δL(x)G(x,x2)]jk = -∫dx Gji(x,x1)[δL(x)G(x,x2)]jk
	# We compute δG_rr(xobs,xsrc) = -∫dx Gjr(x,x1)[δL(x)G(x,x2)]jr

	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs Nν_Gfn

	Gfn_path_obs = Gfn_path_from_source_radius(xobs)
	num_procs_obs = get_numprocs(Gfn_path_obs)

	ν_test_index = searchsortedfirst(ν_arr,ν)
	ν_on_grid = ν_arr[ν_test_index]
	ω = 2π * ν_on_grid

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = ℓ_range[end]

	ν_test_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_range = ν_test_ind:ν_test_ind
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)

	function summodes(ℓ_ωind_iter_on_proc)

		proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,
			1:Nν_Gfn,modes_iter,num_procs)
		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

		proc_id_range_Gobs = get_processor_range_from_split_array(ℓ_arr,
			1:Nν_Gfn,modes_iter,num_procs_obs)
		Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,proc_id_range_Gobs)

		Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1)
		Gobs = zeros_Float64_to_ComplexF64(1:nr,0:1)

		δG = zeros(ComplexF64,nr)
		f_robs_rsrc = zeros(ComplexF64,nr)

		∂ϕ₁Pl_cosχ = dPl(cosχ(xobs,xsrc),lmax=ℓmax).*∂ϕ₁cosχ(xobs,xsrc)

		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

			((ℓ<1) || (ω_ind != ν_test_index )) && continue

			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,:,1:2,1,1)

			read_Gfn_file_at_index!(Gobs,Gfn_fits_files_obs,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,:,1:2,1,1)

			radial_fn_uniform_rotation_firstborn!(f_robs_rsrc,Gsrc,Gobs,ℓ)

			@. δG += (2ℓ+1)/4π * f_robs_rsrc * ∂ϕ₁Pl_cosχ[ℓ]
		end

		close.(values(Gfn_fits_files_src))
		close.(values(Gfn_fits_files_obs))

		-2im*ω*Ω_rot*simps((@. r^2 * ρ * δG),r)
	end

	δG = pmapsum(summodes,modes_iter)
end

# function δGrr_uniform_rotation_firstborn_integrated_over_angle(xobs,xsrc;kwargs...)
# 	# δG_ik(xobs,xsrc) = -∫dx Gij(x1,x)[δL(x)G(x,x2)]jk = -∫dx Gji(x,x1)[δL(x)G(x,x2)]jk
# 	# We compute δG_rr(xobs,xsrc) = -∫dx Gjr(x,x1)[δL(x)G(x,x2)]jr

# 	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
# 	@load(joinpath(Gfn_path_src,"parameters.jld2"),
# 		ν_arr,ℓ_arr,num_procs,Nν_Gfn,ν_start_zeros,
# 		ν_end_zeros,Nν,dω)

# 	Gfn_path_obs = Gfn_path_from_source_radius(xobs)

# 	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
# 	ℓmax = maximum(ℓ_range)

# 	ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)

# 	∂ϕPl_cosχ = dPl(cosχ(xobs,xsrc),lmax=ℓmax).*∂ϕ₁cosχ(xobs,xsrc)

# 	modes_iter = Iterators.product(ℓ_range,ν_ind_range)

# 	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)

# 	function summodes(ℓ_ωind_iter_on_proc)

# 		proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,
# 			1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
# 		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

# 		proc_id_range_Gobs = get_processor_range_from_split_array(ℓ_arr,
# 			1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
# 		Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,proc_id_range_Gobs)

# 		δG = zeros(ComplexF64,nr,Nν_Gfn)
		
# 		Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1)
# 		G00_src = view(Gsrc,:,0)
# 		G10_src = view(Gsrc,:,1)

# 		Gobs = zeros_Float64_to_ComplexF64(1:nr,0:1)
# 		G00_obs = view(Gobs,:,0)
# 		G10_obs = view(Gobs,:,1)

# 		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

# 			(ℓ<1) && continue

# 			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
# 				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,:,1:2,1,1)

# 			read_Gfn_file_at_index!(Gobs,Gfn_fits_files_obs,
# 				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,:,1:2,1,1)

# 			@. δG[:,ω_ind] += (2ℓ+1)/4π * ( G00_src*G00_obs - 
# 						(G00_src*G10_obs + G10_src*G00_obs)/Ω(ℓ,0) +
# 						(ℓ*(ℓ+1)-1)*G10_src*G10_obs/Ω(ℓ,0)^2 ) * ∂ϕPl_cosχ[ℓ]
# 		end

# 		close.(values(Gfn_fits_files_src))
# 		close.(values(Gfn_fits_files_obs))

# 		-2im.*((0:Nν_Gfn-1).*dω).*Ω_rot.*simps((@. r^2 * ρ * δG),r)
# 	end

# 	δG = pmapsum(summodes,modes_iter)
# 	pad_zeros_ν(δG,ν_ind_range,Nν_Gfn,ν_start_zeros,ν_end_zeros)
# end

function δGrrt_uniform_rotation_rotatedwaves(xobs,xsrc;kwargs...)
	G_t = Grrt_rotating_nonrotating(xobs,xsrc;kwargs...)
	G′_t = view(G_t,:,1)
	G0_t = view(G_t,:,2)
	parent(G′_t .- G0_t)
end

function δGrrt_uniform_rotation_rotatedwaves_linearapprox(xobs,xsrc;kwargs...)

	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_path_src,"parameters.jld2") Nt dt

	τ_ind_range = get(kwargs,:τ_ind_range,1:Nt)
	t = vcat(0:div(Nt,2)-1,-div(Nt,2):-1).*dt
	t = t[τ_ind_range]

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)
	
	δGrrt = -Ω_rot * t .* ∂ϕobsGrrt(xobs,xsrc;kwargs...)[τ_ind_range]
	OffsetArray(δGrrt,τ_ind_range)
end

function δGrr_uniform_rotation_rotatedwaves(xobs,xsrc;kwargs...)
	
	# δGrr(x2,x1,ω) = FFT(Grr(x2,x1,t) - G0rr(x2,x1,t))
	# Grr(x2,x1,t) = G0rr(x2-Ωt,x1,t)

	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt
	δGt = δGrrt_uniform_rotation_rotatedwaves(xobs,xsrc;kwargs...)
	fft_t_to_ω(δGt,dt)
end

function δGrr_uniform_rotation_rotatedwaves(xobs,xsrc,ν;kwargs...)
	
	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ν_start_zeros

	ν_ind = searchsortedfirst(ν_arr,ν)

	δGrr_uniform_rotation_rotatedwaves(xobs,xsrc;kwargs...)[ν_start_zeros + ν_ind]
end

function δGrr_uniform_rotation_rotatedwaves_linearapprox(xobs,xsrc;kwargs...)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nν
	ν_ind_range = get(kwargs,:ν_ind_range,1:Nν)
	δGt = δGrrt_uniform_rotation_rotatedwaves_linearapprox(xobs,xsrc;kwargs...)
	fft_t_to_ω(δGt,dt)[ν_ind_range]
end

function δGrr_uniform_rotation_rotatedwaves_linearapprox(xobs,xsrc,ν;kwargs...)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ν_start_zeros
	ν_ind = searchsortedfirst(ν_arr,ν)
	ν_ind_full = ν_ind + ν_start_zeros
	ν_ind_range = ν_ind_full:ν_ind_full
	δGrr_uniform_rotation_rotatedwaves_linearapprox(xobs,xsrc;
							kwargs...,ν_ind_range=ν_ind_range)
end

function δGrr_uniform_rotation_rotatedwaves_linearapprox_FD(xobs,xsrc;kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src,c_scale=get(kwargs,:c_scale,1))
	@load joinpath(Gfn_path_src,"parameters.jld2") dω

	∂ϕG = ∂ϕobsGrrω(xobs,xsrc;kwargs...)
	Ω_rot= get(kwargs,:Ω_rot,20e2/Rsun)

	∂ω∂ϕG = D(size(∂ϕG,1))*∂ϕG ./ dω

	@. -im*Ω_rot*∂ω∂ϕG
end

function δGrr_uniform_rotation_rotatedwaves_linearapprox_FD(xobs,xsrc,ν::Real;kwargs...)
	
	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src,c_scale=get(kwargs,:c_scale,1))

	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr dω ν_start_zeros

	Ω_rot = get(kwargs,:Ω_rot,20e2/Rsun)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)

	ν_test_ind = searchsortedfirst(ν_arr,ν)
	ν_on_grid = ν_arr[ν_test_ind]

	ν_ind_range = max(ν_test_ind-7,1):(ν_test_ind+min(7,ν_test_ind-1))
	ν_match_index = searchsortedfirst(ν_ind_range,ν_test_ind)

	∂ϕG = ∂ϕobsGrrω(xobs,xsrc;
		ν_ind_range=ν_ind_range,kwargs...)[ν_ind_range .+ ν_start_zeros]
	∂ω∂ϕG = D(length(∂ϕG))*∂ϕG ./ dω

	-im*Ω_rot*∂ω∂ϕG[ν_match_index]
end

function divGr_Hansen(xobs::Point3D,xsrc::Point3D;kwargs...)
	# The radial component of divergence can be expressed as 
	# [∇⋅G]ᵣ(x,x′,ω) = ∑ⱼ (2j+1)/4π [∇⋅G]ᵣⱼ(r,r′,ω) Pⱼ(n⋅n′)

	Gfn_path_src = Gfn_path_from_source_radius(xsrc)

	@load joinpath(Gfn_path_src,"parameters.jld2") ℓ_arr num_procs Nν_Gfn ν_arr

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)

	Pl_cosχ = Pl(cosχ(xobs,xsrc),lmax=maximum(ℓ_range))
	r_obs_ind = radial_grid_index(xobs)

	function summodes(ℓ_ωind_iter_on_proc)
		proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,
												ℓ_ωind_iter_on_proc,num_procs)

		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

		Gsrc = zeros_Float64_to_ComplexF64(0:1)
		drGsrc = zeros_Float64_to_ComplexF64(0:0)
		divGrℓ = zeros(ComplexF64,1)
		divGr_arr = zeros(ComplexF64,1)

		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
			
			# Green function about source location
			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,r_obs_ind,1:2,1,1)
			# Derivative of Green function about source location
			read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,r_obs_ind,1:1,1,2)

			divGrℓ .= divG_radial(ℓ,Gsrc,drGsrc)

			@. divGr_arr += (2ℓ+1)/4π * divGrℓ * Pl_cosχ[ℓ]
			
		end

		divGr_arr[r_obs_ind]
	end

	divGr_summed = pmapsum(summodes,modes_iter)
end

function divGr_Hansen(xobs::Point3D,xsrc::Point3D,ν;kwargs...)
	# The radial component of divergence can be expressed as 
	# [∇⋅G]ᵣ(x,x′,ω) = ∑ⱼ (2j+1)/4π [∇⋅G]ᵣⱼ(r,r′,ω) Pⱼ(n⋅n′)

	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load(joinpath(Gfn_path_src,"parameters.jld2"),ν_arr,ν_start_zeros)
	ν_ind = searchsortedfirst(ν_arr,ν)
	divGr_Hansen(xobs,xsrc;kwargs...,ν_ind_range=ν_ind:ν_ind)
end

function divGr_spherical(xobs::Point3D,xsrc::Point3D;kwargs...)
	# The radial component of divergence can be expressed as 
	# [∇⋅G]ᵣ(x,x′,ω) = ∑ⱼ (2j+1)/4π [∇⋅G]ᵣⱼ(r,r′,ω) Pⱼ(n⋅n′)

	Gfn_path_src = Gfn_path_from_source_radius(xsrc)

	@load joinpath(Gfn_path_src,"parameters.jld2") ℓ_arr num_procs Nν_Gfn ν_arr

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
	modes_iter = Iterators.product(ℓ_range,ν_ind_range)

	Pl_matrix = Pl_dPl_d2Pl(cosχ(xobs,xsrc),lmax=maximum(ℓ_range))
	Pl_cosχ = view(Pl_matrix,:,0)
	Pl′_cosχ = view(Pl_matrix,:,1)
	Pl′′_cosχ = view(Pl_matrix,:,2)

	r_obs_ind = radial_grid_index(xobs)

	function summodes(ℓ_ωind_iter_on_proc)
		proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,
												ℓ_ωind_iter_on_proc,num_procs)

		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

		Gsrc = zeros_Float64_to_ComplexF64(0:1)
		drGsrc = zeros_Float64_to_ComplexF64(0:0)
		divGr_arr = zeros(ComplexF64,1)

		r_obs = r[r_obs_ind]

		for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
			
			# Green function about source location
			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,r_obs_ind,1:2,1,1)
			# Derivative of Green function about source location
			read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
				ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs,r_obs_ind,1:1,1,2)

			θ_term = Pl′′_cosχ[ℓ]*∂θ₁cosχ(xobs,xsrc)^2 + Pl′_cosχ[ℓ]*(∂²θ₁cosχ(xobs,xsrc) + ∂θ₁cosχ(xobs,xsrc)*cot(xobs.θ) )
			ϕ_term = Pl′′_cosχ[ℓ]*∂ϕ₁cosχ(xobs,xsrc)*∇ϕ₁cosχ(xobs,xsrc) + Pl′_cosχ[ℓ]*∇ϕ₁∂ϕ₁cosχ(xobs,xsrc)

			@. divGr_arr += (2ℓ+1)/4π *  
					((drGsrc[0]+2/r_obs*Gsrc[0]) * Pl_cosχ[ℓ] + 
					1/r_obs*Gsrc[1]*√2/√(ℓ*(ℓ+1))*(θ_term + ϕ_term))

		end

		divGr_arr[1]
	end

	divGr_summed = pmapsum(summodes,modes_iter)
end

function divGr_spherical(xobs::Point3D,xsrc::Point3D,ν;kwargs...)
	# The radial component of divergence can be expressed as 
	# [∇⋅G]ᵣ(x,x′,ω) = ∑ⱼ (2j+1)/4π [∇⋅G]ᵣⱼ(r,r′,ω) Pⱼ(n⋅n′)

	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load(joinpath(Gfn_path_src,"parameters.jld2"),ν_arr,ν_start_zeros)
	ν_ind = searchsortedfirst(ν_arr,ν)
	divGr_spherical(xobs,xsrc;kwargs...,ν_ind_range=ν_ind:ν_ind)
end

function δGrr_isotropicδc_firstborn_integrated_over_angle_partial(
	ℓ_ωind_iter_on_proc::ProductSplit,
	xobs::Point3D,xsrc::Point3D,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs::Union{Nothing,ParamsGfn}=nothing,
	c_scale=1+1e-5,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=xsrc.r,c_scale=1)
	p_Gobs = read_all_parameters(p_Gobs,r_src=xobs.r,c_scale=1)
	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn = p_Gsrc

	Gfn_path_obs,NGfn_files_obs = p_Gobs.path,p_Gobs.num_procs

	ℓ_range,ν_ind_range = ℓ_ωind_iter_on_proc.iterators
	ℓmax = maximum(ℓ_range)

	Pl_cosχ = Pl(cosχ(xobs,xsrc),lmax=ℓmax)
	
	ϵ = c_scale-1

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
		(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,
		(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1)
	drGsrc = zeros_Float64_to_ComplexF64(1:nr,0:0)
	
	Gobs = zeros_Float64_to_ComplexF64(1:nr,0:1)
	drGobs = zeros_Float64_to_ComplexF64(1:nr,0:0)

	divGsrc = zeros(ComplexF64,nr)
	divGobs = zeros(ComplexF64,nr)

	f = zeros(ComplexF64,nr)
	
	dG = zeros(ComplexF64,nr,Nν_Gfn)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc
		
		# Green function about source location
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:2,1,1)
		# Derivative of Green function about source location
		read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:1,1,2)

		# Green function about observer location
		read_Gfn_file_at_index!(Gobs,Gfn_fits_files_obs,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs,:,1:2,1,1)
		# Derivative of Green function about observer location
		read_Gfn_file_at_index!(drGobs,Gfn_fits_files_obs,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs,:,1:1,1,2)

		radial_fn_isotropic_δc_firstborn!(f,
			Gsrc,drGsrc,divGsrc,Gobs,drGobs,divGobs,ℓ)

		@. dG[:,ω_ind] += (2ℓ+1)/4π * f * Pl_cosχ[ℓ]

		signaltomaster!(progress_channel)
	end

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs))
	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	simps((@. r^2 * ϵ*c * dG),r)
end

function δGrr_isotropicδc_firstborn_integrated_over_angle(xobs,xsrc;kwargs...)

	p_Gsrc = read_all_parameters(r_src=xsrc.r,c_scale=1)
	p_Gobs = read_all_parameters(r_src=xobs.r,c_scale=1)
	c_scale = get(kwargs,:c_scale,1+1e-5)
	@unpack Nν_Gfn,ν_arr,ℓ_arr,ν_start_zeros,ν_end_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	Gω_in_range = pmapsum_timed(δGrr_isotropicδc_firstborn_integrated_over_angle_partial,
		modes_iter,xobs,xsrc,p_Gsrc,p_Gobs,c_scale;
		progress_str="Modes summed in Grrω : ")

	dGrrωFB = pad_zeros_ν(Gω_in_range,ν_ind_range,
			Nν_Gfn,ν_start_zeros,ν_end_zeros)

	@save_to_fits_and_return(dGrrωFB)
end

function δGrr_isotropicδc_firstborn_integrated_over_angle(xobs,xsrc,ν;kwargs...)
	Gfn_path_src = Gfn_path_from_source_radius(xsrc)
	@load(joinpath(Gfn_path_src,"parameters.jld2"),ν_arr,ν_start_zeros)
	ν_ind = searchsortedfirst(ν_arr,ν)

	δGrr_isotropicδc_firstborn_integrated_over_angle(xobs,xsrc;kwargs...,ν_ind_range=ν_ind:ν_ind)[ν_start_zeros+ν_ind]
end

function δGrr_isotropicδc_firstborn_integrated_over_angle_numerical(xobs,xsrc,ν;kwargs...)

	c_scale = get(kwargs,:c_scale,1+1e-5)
	δc = c.*(c_scale-1)

	Gfn_path_src = Gfn_path_from_source_radius(xsrc)

	@load(joinpath(Gfn_path_src,"parameters.jld2"),ℓ_arr)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ℓmax = maximum(ℓ_range)

	θ_full,ϕ_full,_,_,wGL = get_θϕ(ℓmax;kwargs...)
	nθ = length(θ_full)
	dϕ = ϕ_full[2] - ϕ_full[1]; nϕ = length(ϕ_full)

	divG_src = divGr3D(xsrc,ν;kwargs...,c_scale=1,θ=θ_full,ϕ=ϕ_full)
	divG_obs = divGr3D(xobs,ν;kwargs...,c_scale=1,θ=θ_full,ϕ=ϕ_full)

	divG_obs .*= divG_src
	close(divG_src)

	divG_obs .*= @. -ρ*2c*δc

	integrand_θϕ_futures = [@spawnat p int_over_r(divG_obs) 
								for p in procs(divG_obs)]
	integrand_θϕ = reshape(vcat(fetch.(integrand_θϕ_futures)...),nθ,nϕ)
	close(divG_obs)

	sum(wGL.*integrand_θϕ)*dϕ
end

function δGrr_isotropicδc_GminusG0(args...;kwargs...)
	c_scale = get(kwargs,:c_scale,1+1e-5)
	G_c0 = Grrω(args...;kwargs...,c_scale=1) 
	G_c′ = Grrω(args...;kwargs...,c_scale=c_scale)
	dGcmc0 = G_c′ .- G_c0
	@save_to_fits_and_return(dGcmc0)
end

end # module