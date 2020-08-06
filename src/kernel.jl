include("$(@__DIR__)/crosscov.jl")

module Kernel

using Reexport
using StructArrays
@reexport using ..Crosscov

import ..HelioseismicKernels: firstshmodes

import ParallelUtilities: finalize_except_wherewhence

export kernel_ℑu⁺₁₀
export kernel_uₛₜ
export kernel_uₛ₀_rθϕ
export kernel_ψϕₛ₀

export kernel_δc₀₀
export kernel_δcₛₜ

export reinterpret_as_float

l_min(SHModes::SphericalHarmonicModes.SHModeRange) = minimum(l_range(SHModes))
l_max(SHModes::SphericalHarmonicModes.SHModeRange) = maximum(l_range(SHModes))
l_minmax(SHModes::SphericalHarmonicModes.SHModeRange) = (l_min(SHModes), l_max(SHModes))

m_min(SHModes::SphericalHarmonicModes.SHModeRange) = minimum(m_range(SHModes))
m_max(SHModes::SphericalHarmonicModes.SHModeRange) = maximum(m_range(SHModes))
m_minmax(SHModes::SphericalHarmonicModes.SHModeRange) = (m_min(SHModes), m_max(SHModes))

########################################################################################
# Macro to call appropriate 3D method given a 2D one
########################################################################################

macro two_points_on_the_surface(fn)
	callermodule = __module__
	quote
		function $(esc(fn))(m::SeismicMeasurement,nobs1::Point2D,nobs2::Point2D,
			los::los_direction=los_radial();kwargs...)

			r_obs = get(kwargs,:r_obs,r_obs_default)
			xobs1 = Point3D(r_obs,nobs1)
			xobs2 = Point3D(r_obs,nobs2)
			$callermodule.$fn(m,xobs1,xobs2,los;kwargs...)
		end
	end
end

########################################################################################
# Utility function 
########################################################################################

function getkernelmodes(;kwargs...)
	s_min = get(kwargs,:s_min,0) :: Int
	s_max = get(kwargs,:s_max,s_min) :: Int
	t_min = get(kwargs,:t_min,0) :: Int
	t_max = get(kwargs,:t_max,t_min) :: Int

	SHModes = get(kwargs,:SHModes,LM(s_min:s_max,t_min:t_max)) :: LM
end

# remove wrappers to get the underlying array
@inline reinterpret_as_float(arr::AbstractArray) = reinterpret_as_float(parent(arr))
# convert complex arrays to real
@inline function reinterpret_as_float(arr::Array{Complex{T}}) where {T<:Real}
	reinterpret(T,arr)
end
# real arrays may be returned as is
@inline reinterpret_as_float(arr::Array{<:Real}) = arr

function unpackGfnparams(p_Gsrc,r_src,p_Gobs1,r_obs1,p_Gobs2,r_obs2)
	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src)
	p_Gobs1 = read_all_parameters(p_Gobs1,r_src=r_obs1)
	p_Gobs2 = read_all_parameters(p_Gobs2,r_src=r_obs2)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	Gfn_path_obs1,NGfn_files_obs1 =  p_Gobs1.path,p_Gobs1.num_procs
	Gfn_path_obs2,NGfn_files_obs2 = p_Gobs2.path,p_Gobs2.num_procs

	@eponymtuple(Gfn_path_src,NGfn_files_src,
		Gfn_path_obs1,NGfn_files_obs1,
		Gfn_path_obs2,NGfn_files_obs2,
		p_Gsrc,p_Gobs1,p_Gobs2)
end

########################################################################################
# Validation for uniform rotation
########################################################################################

function allocatearrays(::flows,los::los_direction,obs_at_same_height::Bool)
	Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1,srcindG(los)...)
	drGsrc = zeros_Float64_to_ComplexF64(1:nr,0:1,srcindG(los)...)
	Gobs1 = zeros_Float64_to_ComplexF64(1:nr,0:1,srcindG(los)...)
	Gobs2 = Gobs1

	Gobs1_cache = Dict{Int,typeof(Gobs1)}()
	Gobs2_cache = Dict{Int,typeof(Gobs2)}()

	Gparts_r₁ = OffsetVector([zeros(ComplexF64,nr,0:1,srcindG(los)...) for γ=0:1],0:1)
	Gparts_r₂ = obs_at_same_height ? Gparts_r₁ : OffsetVector(
					[zeros(ComplexF64,nr,0:1,srcindG(los)...) for γ=0:1],0:1)

	# This is Gγℓjₒjₛω_α₁0(r,r₁,rₛ) as computed in the paper
	# Stored as Gγℓjₒjₛ_r₁[r,γ,αᵢ] = Gparts_r₁[γ][r,0,αᵢ] + ζ(jₛ,jₒ,ℓ) Gparts_r₁[γ][:,1,αᵢ]
	Gγℓjₒjₛ_r₁ = zeros(ComplexF64,nr,0:1,srcindG(los)...)
	Gγℓjₒjₛ_r₂ = obs_at_same_height ? Gγℓjₒjₛ_r₁ : zero(Gγℓjₒjₛ_r₁)

	# This is given by Hγℓjₒjₛω_α₁α₂(r,r₁,r₂) = conj(Gγℓjₒjₛω_α₁0(r,r₁,rₛ)) * Gjₛω_α₂0(r₂,rₛ)
	# Stored as Hγℓjₒjₛ_r₁r₂[r,γ,α₁,α₂] = conj(Gγℓjₒjₛ_r₁[r,γ,α₁]) * Gα₂r_r₂[α₂]
	Hγℓjₒjₛ_r₁r₂ = zeros(ComplexF64,nr,0:1,srcindG(los)...,obsindG(los)...)
	Hγℓjₒjₛ_r₂r₁ = obs_at_same_height ? Hγℓjₒjₛ_r₁r₂ : zero(Hγℓjₒjₛ_r₁r₂)

	twoimagconjhωHγℓjₒjₛ_r₁r₂ = zeros(nr,0:1,srcindG(los)...,obsindG(los)...)
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁ = zeros(nr,0:1,srcindG(los)...,obsindG(los)...)

	# temporary array to save the γ=-1 component that may be used to compute the γ=1 one
	# the γ = 0 component is also stored here
	temp = zeros(ComplexF64,nr)
	# tempSA = StructArray{ComplexF64}(undef,nr); 
	# fill!(tempSA,zero(eltype(tempSA)))

	# This is Gγℓjₒjₛ_r₁ for γ=1, used for the validation test
	G¹₁jj_r₁ = zeros(ComplexF64,nr,srcindG(los)...)
	G¹₁jj_r₂ = obs_at_same_height ? G¹₁jj_r₁ : zero(G¹₁jj_r₁)

	# Hjₒjₛαβ(r;r₁,r₂,rₛ) = conj(f_αjₒjₛ(r,r₁,rₛ)) Gβ0jₛ(r₂,rₛ)
	# We only use this for the validation case of ℑu⁺, so jₒ = jₛ = j
	H¹₁jj_r₁r₂ = zeros(ComplexF64,nr,srcindG(los)...,obsindG(los)...)
	H¹₁jj_r₂r₁ = obs_at_same_height ? H¹₁jj_r₁r₂ : zero(H¹₁jj_r₁r₂)

	@eponymtuple(Gsrc,Gobs1,Gobs2,drGsrc,
		Gobs1_cache,Gobs2_cache,
		temp,#tempSA,
		Gparts_r₁, Gparts_r₂,
		Gγℓjₒjₛ_r₁,Gγℓjₒjₛ_r₂,
		Hγℓjₒjₛ_r₁r₂, Hγℓjₒjₛ_r₂r₁,
		twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
		twoimagconjhωHγℓjₒjₛ_r₁r₂,
		G¹₁jj_r₁, G¹₁jj_r₂,
		H¹₁jj_r₁r₂, H¹₁jj_r₂r₁)
end

function populatekernelvalidation!(::flows,::los_radial,K::AbstractVector{<:Real},
	(j,ω),∂ϕ₂Pⱼ_cosχ::Real,conjhω,H¹₁jj_r₁r₂,H¹₁jj_r₂r₁,dω,ln₁,ln₂)

	pre = 2*√(3/π) * dω/2π * ω^3 * Powspec(ω) * (2j+1)/4π * ∂ϕ₂Pⱼ_cosχ
	@inbounds for r_ind in axes(K,1)
		K[r_ind] += pre * imag( conjhω * ( H¹₁jj_r₁r₂[r_ind] + conj(H¹₁jj_r₂r₁[r_ind]) ) )
	end
end

function populatekernelvalidation!(::flows,::los_earth,K::AbstractVector{<:Real},
	(j,ω),Y12j::SHArray,conjhω,H¹₁jj_r₁r₂,H¹₁jj_r₂r₁,dω,ln₁,ln₂)

	pre = dω/2π * ω^3 * Powspec(ω) * (-1)^j * √((j*(j+1)*(2j+1))/π)
	ind10 = modeindex(firstshmodes(Y12j),(1,0))

	@inbounds for α₂ in axes(Y12j,2),α₁ in axes(Y12j,1)

		Pʲʲ₁₀_α₁α₂ = pre * imag(ln₁[α₁] * ln₂[α₂] * Y12j[α₁,α₂,ind10])
		iszero(Pʲʲ₁₀_α₁α₂) && continue

		for r_ind in axes(K,1)
			K[r_ind] += Pʲʲ₁₀_α₁α₂ * 2imag(conjhω *
				( H¹₁jj_r₁r₂[r_ind,abs(α₁),abs(α₂)] + conj(H¹₁jj_r₂r₁[r_ind,abs(α₂),abs(α₁)] ) ) )
		end
	end
end

function populatekernelvalidation!(::flows,::los_radial,K::AbstractMatrix{<:Real},
	(j,ω)::Tuple{Int,Real},∂ϕ₂Pⱼ_cosχ::AbstractVector{<:Real},
	conjhω::AbstractVector{ComplexF64},H¹₁jj_r₁r₂,H¹₁jj_r₂r₁,dω,ln₁,ln₂arr)

	pre = 2*√(3/4π) * dω/2π * ω^3 * Powspec(ω) * (2j+1)/4π
	@inbounds for n2ind in axes(K,2)
		pre2 = pre * ∂ϕ₂Pⱼ_cosχ[n2ind]
		conjhω_n2 = conjhω[n2ind]
		for r_ind in axes(K,1)
			K[r_ind,n2ind] +=  pre2 * 2imag( conjhω_n2 * ( H¹₁jj_r₁r₂[r_ind] + conj(H¹₁jj_r₂r₁[r_ind]) ) )
		end
	end
end

function populatekernelvalidation!(::flows,::los_earth,K::AbstractMatrix{<:Real},
	(j,ω),Y12j::AbstractVector{<:SHArray},conjhω,H¹₁jj_r₁r₂,H¹₁jj_r₂r₁,dω,ln₁,l2arr)

	pre = 2 * dω/2π * ω^3 * Powspec(ω) * (-1)^j * √((j*(j+1)*(2j+1))/π)
	ind10 = modeindex(firstshmodes(first(Y12j)),(1,0))

	@inbounds for n2ind in axes(K,2)

		Y12j_n₂ = Y12j[n2ind]
		ln₂ = l2arr[n2ind]
		conjhω_n₂ = conjhω[n2ind]

		for α₂ in axes(Y12j_n₂,2),α₁ in axes(Y12j_n₂,1)

			Pʲʲ₁₀_α₁α₂ = pre * imag(ln₁[α₁] * ln₂[α₂] * Y12j_n₂[α₁,α₂,ind10])
			iszero(Pʲʲ₁₀_α₁α₂) && continue

			for r_ind in axes(K,1)
				K[r_ind,n2ind] += Pʲʲ₁₀_α₁α₂ * imag(conjhω_n₂ * 
					( H¹₁jj_r₁r₂[r_ind,abs(α₁),abs(α₂)] + conj(H¹₁jj_r₂r₁[r_ind,abs(α₂),abs(α₁)] ) ) )
			end
		end
	end
end

function kernel_ℑu⁺₁₀_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_direction,hω,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()
	
	@unpack p_Gsrc,p_Gobs1,p_Gobs2,Gfn_path_src,NGfn_files_src,
	Gfn_path_obs1,NGfn_files_obs1,Gfn_path_obs2,NGfn_files_obs2 = 
		unpackGfnparams(p_Gsrc,r_src,p_Gobs1,xobs1.r,p_Gobs2,xobs2.r)

	@unpack ℓ_arr,ω_arr,Nν_Gfn,dω = p_Gsrc

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))
	obs_at_same_height = r₁_ind == r₂_ind

	Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2 = 
			Gfn_fits_files((Gfn_path_src,Gfn_path_obs1,Gfn_path_obs2),
				((ℓ_arr,1:Nν_Gfn)),ℓ_ωind_iter_on_proc,
				(NGfn_files_src,NGfn_files_obs1,NGfn_files_obs2))

	K = zeros(nr)

	arrs = allocatearrays(flows(),los,obs_at_same_height)
	@unpack Gsrc,Gobs1,Gobs2, G¹₁jj_r₁, G¹₁jj_r₂, H¹₁jj_r₁r₂, H¹₁jj_r₂r₁ = arrs

	ℓ_range = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	Y12 = computeY₁₀(los,xobs1,xobs2,ℓ_range)

	# covariant components
	l1,l2 = line_of_sight_covariant(xobs1,xobs2,los)

	for (j,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]
		conjhω = conj(hω[ω_ind])

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(j,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),1)

		Gγr_r₁_rsrc = αrcomp(Gsrc,r₁_ind,los)
		Gγr_r₂_rsrc = αrcomp(Gsrc,r₂_ind,los)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(j,ω_ind),NGfn_files_obs1,:,1:2,srcindFITS(los),1)

		radial_fn_uniform_rotation_firstborn!(G¹₁jj_r₁,Gsrc,Gobs1,j,los)
		Hjₒjₛω!(H¹₁jj_r₁r₂,G¹₁jj_r₁,Gγr_r₂_rsrc)

		if !obs_at_same_height
			# Green function about receiver location
			read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
			(ℓ_arr,1:Nν_Gfn),(j,ω_ind),NGfn_files_obs2,:,1:2,srcindFITS(los),1)

			radial_fn_uniform_rotation_firstborn!(G¹₁jj_r₂,Gsrc,Gobs2,j,los)
			Hjₒjₛω!(H¹₁jj_r₂r₁,G¹₁jj_r₂,Gγr_r₁_rsrc)
		end

		populatekernelvalidation!(flows(),los,K,(j,ω),
			Y12[j],conjhω,H¹₁jj_r₁r₂,H¹₁jj_r₂r₁,dω,l1,l2)

		signaltomaster!(progress_channel)
	end

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2))
	
	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))
	
	return @. (ρ/r)*K
end

function kernel_ℑu⁺₁₀_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	nobs1::Point2D,nobs2_arr::Vector{<:Point2D},los::los_direction,hω,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=p_Gobs1,
	r_src=r_src_default,r_obs=r_obs_default,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	hω = permutedims(hω) #  convert to (n2,ω)

	r_obs_ind = radial_grid_index(r_obs)

	@unpack p_Gsrc,p_Gobs1,Gfn_path_src,NGfn_files_src = 
		unpackGfnparams(p_Gsrc,r_src,p_Gobs1,r_obs,p_Gobs2,r_obs)

	Gfn_path_obs,NGfn_files_obs = p_Gobs1.path,p_Gobs1.num_procs

	@unpack ℓ_arr,ω_arr,Nν_Gfn,dω = p_Gsrc

	Gfn_fits_files_src,Gfn_fits_files_obs = 
			Gfn_fits_files((Gfn_path_src,Gfn_path_obs),
				(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,
				(NGfn_files_src,NGfn_files_obs))

	K = zeros(nr,length(nobs2_arr))

	arrs = allocatearrays(flows(),los,true)
	@unpack Gsrc, H¹₁jj_r₁r₂, G¹₁jj_r₁, Gobs1 = arrs

	ℓ_range = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	Y12 = computeY₁₀(los,nobs1,nobs2_arr,ℓ_range)

	# covariant components
	l1,l2arr = line_of_sight_covariant(nobs1,nobs2_arr,los)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]
		conjhω = conj(@view hω[:,ω_ind])

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),1)

		G_r₂_rsrc = αrcomp(Gsrc,r_obs_ind,los)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs,:,1:2,srcindFITS(los),1)

		radial_fn_uniform_rotation_firstborn!(G¹₁jj_r₁,Gsrc,Gobs1,ℓ,los)
		Hjₒjₛω!(H¹₁jj_r₁r₂,G¹₁jj_r₁,G_r₂_rsrc)

		populatekernelvalidation!(flows(),los,K,(ℓ,ω),
			Y12[ℓ],conjhω,H¹₁jj_r₁r₂,H¹₁jj_r₁r₂,dω,l1,l2arr)

		signaltomaster!(progress_channel)
	end

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs))
	
	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))
	
	return @. (ρ/r)*K
end

# Feeder function

function _K_ℑu⁺₁₀(m::SeismicMeasurement,xobs1,xobs2,los::los_direction,args...;kwargs...)
	r_src,r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc,p_Gobs1,p_Gobs2 = read_parameters_for_points(xobs1,xobs2;kwargs...)
	@unpack ν_arr,ℓ_arr,ν_start_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	# Compute the appropriate window function depending on the parameter of interest
	hω_arr = get(kwargs,:hω) do
		hω(m,xobs1,xobs2,los;kwargs...,print_timings=false)
	end

	# Use the window function to compute the kernel
	pmapsum_timed(kernel_ℑu⁺₁₀_partial,modes_iter,
		xobs1,xobs2,los,args...,hω_arr,p_Gsrc,p_Gobs1,p_Gobs2,r_src,r_obs;
		progress_str="Modes summed in kernel : ",
		print_timings=get(kwargs,:print_timings,false))
end

# Traveltimes

kernelfilenameℑu⁺₁₀(::TravelTimes,::los_radial) = "K_δτ_ℑu⁺₁₀.fits"
kernelfilenameℑu⁺₁₀(::TravelTimes,::los_earth) = "K_δτ_ℑu⁺₁₀_los.fits"
kernelfilenameℑu⁺₁₀(::Amplitudes,::los_radial) = "K_A_ℑu⁺₁₀.fits"
kernelfilenameℑu⁺₁₀(::Amplitudes,::los_earth) = "K_A_ℑu⁺₁₀_los.fits"

function kernel_ℑu⁺₁₀(m::SeismicMeasurement,xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)
	
	K = _K_ℑu⁺₁₀(m,xobs1,xobs2,los;kwargs...)
	
	if get(kwargs,:save,true)
		filename = kernelfilenameℑu⁺₁₀(m,los)
		FITS(filename,"w") do f
			write(f,reinterpret_as_float(K))
		end
	end

	return K
end

@two_points_on_the_surface kernel_ℑu⁺₁₀

########################################################################################
# Kernels for flow velocity
########################################################################################

@inline function Kterm!(::los_radial,K,vindK,modeind,vindT,
	pre,T₂₁,P₂₁,T₁₂,P₁₂)

	P₂₁ *= pre
	P₁₂ *= pre
	@inbounds for r_ind in axes(K,1)
		K[r_ind,vindK,modeind] += (
			T₂₁[r_ind,vindT] * P₂₁ -
			T₁₂[r_ind,vindT] * P₁₂ )
	end
end
@inline function tempterm!(::los_radial,temp,vindT,
	pre,T₂₁,P₂₁,T₁₂,P₁₂)

	P₂₁ *= pre
	P₁₂ *= pre
	@inbounds for r_ind in axes(temp,1)
		temp[r_ind] += (
			T₂₁[r_ind,vindT] * P₂₁ -
			T₁₂[r_ind,vindT] * P₁₂ )
	end
end
@inline function addtemptoK!(K,vind,modeind,temp,phase = one(eltype(temp)))
	@inbounds for r_ind in axes(K,1)
		K[r_ind,vind,modeind] += phase * temp[r_ind]
	end
end
@inline function Kterm!(::los_earth,K,vindK,modeind,vindT,
	pre,T₂₁,P₂₁,T₁₂,P₁₂,α₁,α₂)

	P₂₁ *= pre
	P₁₂ *= pre
	@inbounds for r_ind in axes(K,1)
		K[r_ind,vindK,modeind] += (
		T₂₁[r_ind,vindT,α₁,α₂] * P₂₁ -
		T₁₂[r_ind,vindT,α₁,α₂] * P₁₂ )
	end
end
@inline function tempterm!(::los_earth,temp,vindT,
	pre,T₂₁,P₂₁,T₁₂,P₁₂,α₁,α₂)

	P₂₁ *= pre
	P₁₂ *= pre
	@inbounds for r_ind in axes(temp,1)
		temp[r_ind] += (
		T₂₁[r_ind,vindT,α₁,α₂] * P₂₁ -
		T₁₂[r_ind,vindT,α₁,α₂] * P₁₂ )
	end
end

function populatekernel!(::flows,los::los_radial,K::OffsetArray{T,3,Array{T,3}},
	SHModes,Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
	(l,m),(jₛ,jₒ,ω),
	twoimagconjhωHγℓjₒjₛ_r₁r₂,
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
	ln₁,ln₂,pre⁰,pre⁺,phase,temp,
	Dlmn::Nothing) where {T<:Complex}

	# Dlmn being nothing indicates an identity D-matrix
	
	mode_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂),(l,m))

	Pʲₒʲₛₗₘ_₀₀_n₁n₂ = Yjₒjₛ_lm_n₁n₂[mode_ind]
	Pʲₒʲₛₗₘ_₀₀_n₂n₁ = Yjₒjₛ_lm_n₂n₁[mode_ind]

	iszero(Pʲₒʲₛₗₘ_₀₀_n₁n₂) && iszero(Pʲₒʲₛₗₘ_₀₀_n₂n₁) && return

	mode_ind_K = modeindex(SHModes,(l,m))

	minusindK, zeroindK, plusindK = axes(K,2)
	zeroindT, plusindT = axes(twoimagconjhωconjHγℓjₒjₛ_r₂r₁,2)

	anyzeromom = l == 0 || jₛ == 0

	if iseven(jₒ + jₛ + l)
		Kterm!(los, K, zeroindK, mode_ind_K, zeroindT, pre⁰,
			twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_₀₀_n₂n₁,
			twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_₀₀_n₁n₂)
	end

	# l=0 only has γ=0
	anyzeromom && return
	fill!(temp,zero(eltype(temp)))

	tempterm!(los, temp, plusindT, pre⁺,
		twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_₀₀_n₂n₁,
		twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_₀₀_n₁n₂)

	addtemptoK!(K,plusindK,mode_ind_K,temp)
	addtemptoK!(K,minusindK,mode_ind_K,temp,phase)
end

function populatekernel!(::flows,los::los_radial,K::OffsetArray{T,3,Array{T,3}},
	SHModes,Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
	(l,m),(jₛ,jₒ,ω),
	twoimagconjhωHγℓjₒjₛ_r₁r₂,
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
	ln₁,ln₂,pre⁰,pre⁺,phase,temp,
	Dlmn::Union{WignerDMatrix,WignerdMatrix}) where {T<:Complex}
	
	mode_ind_K = modeindex(SHModes,(l,m))

	minusindK, zeroindK, plusindK = axes(K,2)
	zeroindT, plusindT = axes(twoimagconjhωconjHγℓjₒjₛ_r₂r₁,2)

	anyzeromom = l == 0 || jₛ == 0

	Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₁n₂ = zero(ComplexF64)
	Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₂n₁ = zero(Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₁n₂)

	# Compute ∑ Dm′m Plm′	
	for m′ in 0:l
		mode_ind_lm′ = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂),(l,m′))
		Pʲₒʲₛₗₘ′_₀₀_n₁n₂ = Yjₒjₛ_lm_n₁n₂[mode_ind_lm′]
		Pʲₒʲₛₗₘ′_₀₀_n₂n₁ = Yjₒjₛ_lm_n₂n₁[mode_ind_lm′]

		iszero(Pʲₒʲₛₗₘ′_₀₀_n₁n₂) && iszero(Pʲₒʲₛₗₘ′_₀₀_n₂n₁) && continue

		Dlm′m = Dlmn[m′,m]
		if !iszero(Dlm′m)
			Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₁n₂ += Dlm′m * Pʲₒʲₛₗₘ′_₀₀_n₁n₂
			Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₂n₁ += Dlm′m * Pʲₒʲₛₗₘ′_₀₀_n₂n₁
		end

		# negative m′
		if m′ > 0
			Dl₋m′m = Dlmn[-m′,m]
		
			if !iszero(Dl₋m′m)
				conjphase = (-1)^(jₛ + jₒ + l + m′)
				Pʲₒʲₛₗ₋ₘ′_₀₀_n₁n₂ = conjphase * conj(Pʲₒʲₛₗₘ′_₀₀_n₁n₂)
				Pʲₒʲₛₗ₋ₘ′_₀₀_n₂n₁ = conjphase * conj(Pʲₒʲₛₗₘ′_₀₀_n₂n₁)

				Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₁n₂ += Dl₋m′m * Pʲₒʲₛₗ₋ₘ′_₀₀_n₁n₂
				Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₂n₁ += Dl₋m′m * Pʲₒʲₛₗ₋ₘ′_₀₀_n₂n₁
			end
		end
	end

	if iseven(jₒ + jₛ + l)
		Kterm!(los, K, zeroindK, mode_ind_K, zeroindT, pre⁰,
			twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₂n₁,
			twoimagconjhωHγℓjₒjₛ_r₁r₂, Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₁n₂)
	end

	# l=0 only has γ=0
	anyzeromom && return

	fill!(temp,zero(eltype(temp)))
	
	tempterm!(los, temp, plusindT, pre⁺,
		twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₂n₁,
		twoimagconjhωHγℓjₒjₛ_r₁r₂, Dₘ′ₘPʲₒʲₛₗₘ′_₀₀_n₁n₂)

	addtemptoK!(K,plusindK,mode_ind_K,temp)
	addtemptoK!(K,minusindK,mode_ind_K,temp,phase)
end

function populatekernel!(::flows,los::los_radial,K::StructArray{<:Complex,3},
	SHModes,Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
	(l,m),(jₛ,jₒ,ω),
	twoimagconjhωHγℓjₒjₛ_r₁r₂,
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
	ln₁,ln₂,pre⁰,pre⁺,phase,temp,
	Dlmn::Nothing)

	# Dlmn being nothing indicates an identity D-matrix
	
	mode_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂),l,m)

	Pʲₒʲₛₗₘ_₀₀_n₁n₂ = Yjₒjₛ_lm_n₁n₂[mode_ind]
	Pʲₒʲₛₗₘ_₀₀_n₂n₁ = Yjₒjₛ_lm_n₂n₁[mode_ind]

	iszero(Pʲₒʲₛₗₘ_₀₀_n₁n₂) && iszero(Pʲₒʲₛₗₘ_₀₀_n₂n₁) && return

	mode_ind_K = modeindex(SHModes,l,m)

	minusindK, zeroindK, plusindK = axes(K,2)
	zeroindT, plusindT = axes(twoimagconjhωconjHγℓjₒjₛ_r₂r₁,2)

	if iseven(jₒ + jₛ + l)
		
		Kterm!(los_radial(),K.re, zeroindK, mode_ind_K, zeroindT, 
			pre⁰, twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_₀₀_n₂n₁.re,
			twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_₀₀_n₁n₂.re)

		Kterm!(los_radial(),K.im, zeroindK, mode_ind_K, zeroindT,
			pre⁰, twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_₀₀_n₂n₁.im,
			twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_₀₀_n₁n₂.im)
	end

	# l=0 only has γ=0
	(l == 0 || jₛ == 0) && return

	tempterm!(los_radial(),temp.re, plusindT, 
		pre⁺, twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_₀₀_n₂n₁.re,
		twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_₀₀_n₁n₂.re)

	tempterm!(los_radial(),temp.im, plusindT, 
		pre⁺, twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_₀₀_n₂n₁.im,
		twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_₀₀_n₁n₂.im)

	addtemptoK!(K.re,plusindK,mode_ind_K,temp.re)
	addtemptoK!(K.im,plusindK,mode_ind_K,temp.im)

	addtemptoK!(K.re,minusindK,mode_ind_K,temp.re,phase)
	addtemptoK!(K.im,minusindK,mode_ind_K,temp.im,phase)
end

function populatekernel!(::flows,::los_earth,K::OffsetArray{T,3,Array{T,3}},
	SHModes,Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
	(l,m),(jₛ,jₒ,ω),
	twoimagconjhωHγℓjₒjₛ_r₁r₂,
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
	ln₁,ln₂,pre⁰,pre⁺,phase,temp,
	Dlmn::Nothing) where {T<:Complex}

	# Dlmn being nothing indicates an identity D-matrix

	mode_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂),(l,m))
	mode_ind_K = modeindex(SHModes,(l,m))

	# temp holds the tangential (+) component
	fill!(temp,zero(eltype(temp)))

	tangentialzero = true # true if nothing is added to the tangential components

	@inbounds for α₂ = axes(Yjₒjₛ_lm_n₁n₂,2), α₁ = axes(Yjₒjₛ_lm_n₁n₂,1)
		
		Pʲₒʲₛₗₘ_α₁α₂_n₁n₂ = ln₁[α₁] * ln₂[α₂] * Yjₒjₛ_lm_n₁n₂[α₁,α₂,mode_ind]
		Pʲₒʲₛₗₘ_α₁α₂_n₂n₁ = ln₂[α₁] * ln₁[α₂] * Yjₒjₛ_lm_n₂n₁[α₁,α₂,mode_ind]

		iszero(Pʲₒʲₛₗₘ_α₁α₂_n₁n₂) && iszero(Pʲₒʲₛₗₘ_α₁α₂_n₂n₁) && continue

		if (0 in axes(K,2)) && iseven(jₒ + jₛ + l)
			for r_ind in axes(K,1)
				K[r_ind,0,mode_ind_K] += pre⁰ * (
				twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind,0,abs(α₁),abs(α₂)] * Pʲₒʲₛₗₘ_α₁α₂_n₂n₁ -
				twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind,0,abs(α₁),abs(α₂)] * Pʲₒʲₛₗₘ_α₁α₂_n₁n₂ )
			end
		end

		((l == 0) || (jₛ == 0)) && continue

		if ((-1 in axes(K,2)) || (1 in axes(K,2)))
			for r_ind in eachindex(temp)
				temp[r_ind] += pre⁺ * (
				twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind,1,abs(α₁),abs(α₂)] * Pʲₒʲₛₗₘ_α₁α₂_n₂n₁ -
				twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind,1,abs(α₁),abs(α₂)] * Pʲₒʲₛₗₘ_α₁α₂_n₁n₂ )
			end
			tangentialzero = false
		end
	end

	tangentialzero && return

	if 1 in axes(K,2)
		@inbounds for r_ind in axes(K,1)
			K[r_ind,1,mode_ind_K] += temp[r_ind]
		end
	end

	if -1 in axes(K,2)
		@inbounds for r_ind in axes(K,1)
			K[r_ind,-1,mode_ind_K] += phase * temp[r_ind]
		end
	end
end

function populatekernel!(::flows,::los_earth,K::StructArray{<:Complex,3},
	SHModes,Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
	(l,m),(jₛ,jₒ,ω),
	twoimagconjhωHγℓjₒjₛ_r₁r₂,
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
	ln₁,ln₂,pre⁰,pre⁺,phase,temp,
	Dlmn::Nothing)

	# Dlmn being nothing indicates an identity D-matrix

	mode_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂),l,m)
	mode_ind_K = modeindex(SHModes,l,m)

	# temp holds the tangential (+) component

	evenjsum = iseven(jₒ + jₛ + l)
	anyzeromom = l == 0 || jₛ == 0

	fill!(temp,zero(eltype(temp)))

	minusindK, zeroindK, plusindK = axes(K,2)
	zeroindT, plusindT = axes(twoimagconjhωconjHγℓjₒjₛ_r₂r₁,2)

	@inbounds for α₂ = axes(Yjₒjₛ_lm_n₁n₂,2), α₁ = axes(Yjₒjₛ_lm_n₁n₂,1)
		
		Pʲₒʲₛₗₘ_α₁α₂_n₁n₂ = ln₁[α₁] * ln₂[α₂] * Yjₒjₛ_lm_n₁n₂[α₁,α₂,mode_ind]
		Pʲₒʲₛₗₘ_α₁α₂_n₂n₁ = ln₂[α₁] * ln₁[α₂] * Yjₒjₛ_lm_n₂n₁[α₁,α₂,mode_ind]

		iszero(Pʲₒʲₛₗₘ_α₁α₂_n₁n₂) && iszero(Pʲₒʲₛₗₘ_α₁α₂_n₂n₁) && continue

		if evenjsum
			Kterm!(los_earth(),K.re, zeroindK, mode_ind_K, zeroindT, 
				pre⁰, twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_α₁α₂_n₂n₁.re,
				twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_α₁α₂_n₁n₂.re, abs(α₁), abs(α₂))

			Kterm!(los_earth(),K.im, zeroindK, mode_ind_K, zeroindT,
				pre⁰, twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_α₁α₂_n₂n₁.im,
				twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_α₁α₂_n₁n₂.im, abs(α₁), abs(α₂))
		end

		anyzeromom && continue

		tempterm!(los_earth(),temp.re, plusindT, 
			pre⁺, twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_α₁α₂_n₂n₁.re,
			twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_α₁α₂_n₁n₂.re, abs(α₁), abs(α₂))

		tempterm!(los_earth(),temp.im, plusindT, 
			pre⁺, twoimagconjhωconjHγℓjₒjₛ_r₂r₁, Pʲₒʲₛₗₘ_α₁α₂_n₂n₁.im,
			twoimagconjhωHγℓjₒjₛ_r₁r₂, Pʲₒʲₛₗₘ_α₁α₂_n₁n₂.im, abs(α₁), abs(α₂))
	end

	anyzeromom && return

	addtemptoK!(K.re,plusindK,mode_ind_K,temp.re)
	addtemptoK!(K.im,plusindK,mode_ind_K,temp.im)

	addtemptoK!(K.re,minusindK,mode_ind_K,tempSA.re,phase)
	addtemptoK!(K.im,minusindK,mode_ind_K,tempSA.im,phase)
end

function populatekernelrθϕl0!(::flows,::los_radial,K::AbstractArray{<:Real,3},
	SHModes,Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
	(l,_),(jₛ,jₒ,ω),conjhω,Hγℓjₒjₛ_r₁r₂,Hγℓjₒjₛ_r₂r₁,
	ln₁,ln₂,pre⁰,pre⁺,phase,temp)
	
	lm_ind_K = modeindex(SHModes,(l,0))
	lm_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂),(l,0))
	@inbounds Pʲₒʲₛₗₘ_₀₀_n₁n₂ = Yjₒjₛ_lm_n₁n₂[lm_ind]
	@inbounds Pʲₒʲₛₗₘ_₀₀_n₂n₁ = Yjₒjₛ_lm_n₂n₁[lm_ind]

	iszero(Pʲₒʲₛₗₘ_₀₀_n₁n₂) && iszero(Pʲₒʲₛₗₘ_₀₀_n₂n₁) && return

	if (1 in axes(K,2)) && iseven(jₒ + jₛ + l)
		@inbounds for r_ind in axes(K,1)
			K[r_ind,1,lm_ind_K] += pre⁰ * (
				2imag(conjhω * conj(Hγℓjₒjₛ_r₂r₁[r_ind,0]) ) * real(Pʲₒʲₛₗₘ_₀₀_n₂n₁) -
				2imag(conjhω * Hγℓjₒjₛ_r₁r₂[r_ind,0] ) * real(Pʲₒʲₛₗₘ_₀₀_n₁n₂) )
		end
	end

	# l=0 only has γ=0
	((l == 0) || (jₛ == 0)) && return

	if ((2 in axes(K,2)) || (3 in axes(K,2)))
		@inbounds for r_ind in eachindex(temp)
			temp[r_ind] = pre⁺ * (
				2imag(conjhω * conj(Hγℓjₒjₛ_r₂r₁[r_ind,1]) ) * Pʲₒʲₛₗₘ_₀₀_n₂n₁ -
				2imag(conjhω * Hγℓjₒjₛ_r₁r₂[r_ind,1] ) * Pʲₒʲₛₗₘ_₀₀_n₁n₂ )
		end
	end

	if 2 in axes(K,2) && iseven(jₒ + jₛ + l)
		@inbounds for r_ind in axes(K,1)
			K[r_ind,2,lm_ind_K] += 2real(temp[r_ind])
		end
	end

	if 3 in axes(K,2) && isodd(jₒ + jₛ + l)
		@inbounds for r_ind in axes(K,1)
			K[r_ind,3,lm_ind_K] += -2imag(temp[r_ind])
		end
	end
end

function populatekernelrθϕl0!(::flows,::los_earth,K::AbstractArray{<:Real,3},
	SHModes,Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
	(l,_),(jₛ,jₒ,ω),conjhω,Hγℓjₒjₛ_r₁r₂,Hγℓjₒjₛ_r₂r₁,
	ln₁,ln₂,pre⁰,pre⁺,phase,temp)

	lm_ind_K = modeindex(SHModes,(l,0))
	lm_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂),(l,0))

	# temp holds the tangential (+) component
	if ((2 in axes(K,2)) || (3 in axes(K,2)))
		fill!(temp,zero(eltype(temp)))
	end

	tangentialzero = true
	@inbounds for α₂ = axes(Yjₒjₛ_lm_n₁n₂,2), α₁ = axes(Yjₒjₛ_lm_n₁n₂,1)
		
		Pʲₒʲₛₗₘ_α₁α₂_n₁n₂ = ln₁[α₁] * ln₂[α₂] * Yjₒjₛ_lm_n₁n₂[α₁,α₂,lm_ind]
		Pʲₒʲₛₗₘ_α₁α₂_n₂n₁ = ln₂[α₁] * ln₁[α₂] * Yjₒjₛ_lm_n₂n₁[α₁,α₂,lm_ind]

		iszero(Pʲₒʲₛₗₘ_α₁α₂_n₁n₂) && iszero(Pʲₒʲₛₗₘ_α₁α₂_n₂n₁) && continue

		if (1 in axes(K,2)) && iseven(jₒ + jₛ + l)
			for r_ind in axes(K,1)
				K[r_ind,1,lm_ind_K] += pre⁰ * (
				2imag(conjhω * conj(Hγℓjₒjₛ_r₂r₁[r_ind,0,abs(α₁),abs(α₂)]) ) * real(Pʲₒʲₛₗₘ_α₁α₂_n₂n₁) -
				2imag(conjhω * Hγℓjₒjₛ_r₁r₂[r_ind,0,abs(α₁),abs(α₂)] ) * real(Pʲₒʲₛₗₘ_α₁α₂_n₁n₂) )
			end
		end

		if ((2 in axes(K,2)) || (3 in axes(K,2))) && (l > 0) && (jₛ > 0)
			for r_ind in axes(temp,1)
				temp[r_ind] += pre⁺ * (
				2imag(conjhω * conj(Hγℓjₒjₛ_r₂r₁[r_ind,1,abs(α₁),abs(α₂)]) ) * Pʲₒʲₛₗₘ_α₁α₂_n₂n₁ -
				2imag(conjhω * Hγℓjₒjₛ_r₁r₂[r_ind,1,abs(α₁),abs(α₂)] ) * Pʲₒʲₛₗₘ_α₁α₂_n₁n₂ )
			end
			tangentialzero = false
		end
	end

	tangentialzero && return

	if 2 in axes(K,2) && iseven(jₒ + jₛ + l)
		@inbounds for r_ind in axes(K,1)
			K[r_ind,2,lm_ind_K] += 2real(temp[r_ind])
		end
	end

	if 3 in axes(K,2) && isodd(jₒ + jₛ + l)
		@inbounds for r_ind in axes(K,1)
			K[r_ind,3,lm_ind_K] += -2imag(temp[r_ind])
		end
	end
end

@inline function mulprefactor!(K::OffsetArray{T,3,Array{T,3}}) where {T<:Complex}
	@inbounds for lmind in axes(K,3), vind in axes(K,2), r_ind in axes(K,1)
		K[r_ind,vind,lmind] *= ρ[r_ind]
	end
	
	@inbounds for lmind in axes(K,3), vind in axes(K,2)
		vind == 0 && continue
		for r_ind in axes(K,1)
			K[r_ind,vind,lmind] /= r[r_ind]
		end
	end
end

@inline function mulprefactor!(K::StructArray{<:Complex,3})
	@inbounds for lmind in axes(K,3), vind in axes(K,2), r_ind in axes(K,1)
		K.re[r_ind,vind,lmind] *= ρ[r_ind]
	end

	@inbounds for lmind in axes(K,3), vind in axes(K,2), r_ind in axes(K,1)
		K.im[r_ind,vind,lmind] *= ρ[r_ind]
	end
	
	@inbounds for lmind in axes(K,3), vind in axes(K,2)
		vind == 2 && continue # radial component
		for r_ind in axes(K,1)
			K.re[r_ind,vind,lmind] /= r[r_ind]
		end
		for r_ind in axes(K,1)
			K.im[r_ind,vind,lmind] /= r[r_ind]
		end
	end
end

WignerD.WignerDMatrix(::Type, ::Integer, ::Nothing, ::Nothing, ::Nothing) = nothing

maybeallmodes(SHModes,::Nothing) = SHModes
maybeallmodes(SHModes, ::Tuple) = LM(l_range(SHModes), ZeroTo)

function kernel_uₛₜ_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_direction,SHModes,hω_arr,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=r_obs_default,
	K_components::UnitRange{Int}=-1:1,
	eulerangles::Union{Nothing,NTuple{3,Real}}=nothing,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	@unpack p_Gsrc,p_Gobs1,p_Gobs2,Gfn_path_src,NGfn_files_src,Gfn_path_obs1,NGfn_files_obs1,
	Gfn_path_obs2,NGfn_files_obs2 = unpackGfnparams(p_Gsrc,r_src,p_Gobs1,xobs1.r,p_Gobs2,xobs2.r)

	@unpack ℓ_arr,ω_arr,Nν_Gfn,dω = p_Gsrc

	s_range = l_range(SHModes)
	s_max = maximum(s_range)
	
	αrot, βrot, γrot = unpackeuler(eulerangles)

	Dlmn_arr = OffsetVector(
		[WignerDMatrix(ComplexF64, s, αrot, βrot, γrot) for s in s_range],s_range)

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))
	obs_at_same_height = r₁_ind == r₂_ind

	ℓ_range_proc = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)

	# Get a list of all modes that will be accessed.
	# This can be used to open the fits files before the loops begin.
	# This will cut down on FITS IO costs
	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	# Gℓ′ω(r,robs) files
	Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_obs1,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,s_max,NGfn_files_obs1)
	# Gℓ′ω(r,robs) files
	Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_obs2,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,s_max,NGfn_files_obs2)

	K = zeros(ComplexF64,nr,-1:1,length(SHModes)) # Kγₗₘ(r,x₁,x₂)

	arrs = allocatearrays(flows(),los,obs_at_same_height);
	@unpack Gsrc,drGsrc,Gparts_r₁,Gparts_r₂,
	Gγℓjₒjₛ_r₁,Gγℓjₒjₛ_r₂,Hγℓjₒjₛ_r₁r₂,Hγℓjₒjₛ_r₂r₁,temp = arrs;
	@unpack twoimagconjhωconjHγℓjₒjₛ_r₂r₁,twoimagconjhωHγℓjₒjₛ_r₁r₂ = arrs;

	jₒjₛ_allmodes = L2L1Triangle(ℓ_range_proc,s_max,ℓ_arr)
	jₒrange = l2_range(jₒjₛ_allmodes)
	@unpack Gobs1_cache,Gobs2_cache = arrs
	for jₒ in jₒrange
		Gobs1_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(Gsrc)...)
		if !obs_at_same_height
			Gobs2_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(Gsrc)...)
		end
	end 

	SHModes_new = maybeallmodes(SHModes, eulerangles)
	@timeit localtimer "BiPoSH" begin
		Y_lm_jrange_n1n2,Y_lm_jrange_n2n1 = BiPoSH_n1n2_n2n1(
			shtype(los),VSHtype(xobs1,xobs2),xobs1,xobs2,
			SHModes_new,jₒjₛ_allmodes)
	end #localtimer

	# Line of sight (if necessary)
	l1,l2 = line_of_sight_covariant(xobs1,xobs2,los)

	C = zeros(0:1,0:s_max,l2_range(jₒjₛ_allmodes),ℓ_range_proc)

	@timeit localtimer "CG" begin
	@inbounds for jₛ in axes(C,4), jₒ in axes(C,3), ℓ in axes(C,2)
		
		C[0,ℓ,jₒ,jₛ] = if iseven(jₒ+jₛ+ℓ)
							clebschgordan(Float64,jₒ,0,jₛ,0,ℓ,0)
						else
							zero(Float64)
						end

		C[1,ℓ,jₒ,jₛ] = if ℓ > 0 && jₛ > 0
							clebschgordan(Float64,jₒ,0,jₛ,1,ℓ,1)
						else
							zero(Float64)
						end
	end
	end # timer

	C⁰,C⁺ = zero(Float64),zero(Float64)
	phase = 1
	pre⁰,pre⁺ = zero(Float64), zero(Float64)

	ω_ind_prev = first(ℓ_ωind_iter_on_proc)[2] - 1

	# Loop over the Greenfn files
	for (jₛ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]
		h_ω = hω_arr[ω_ind]
		conjhω = conj(h_ω)
		dωω³Pω = dω/2π * ω^3 * Powspec(ω)
		Ωjₛ0 = Ω(jₛ,0)

		@timeit localtimer "FITS" begin
		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
		(ℓ_arr,1:Nν_Gfn),(jₛ,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
		(ℓ_arr,1:Nν_Gfn),(jₛ,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),2)
		end # timer

		Gα₂r_r₁_rₛ = αrcomp(Gsrc,r₁_ind,los)
		Gα₂r_r₂_rₛ = αrcomp(Gsrc,r₂_ind,los)

		for jₒ in l2_range(jₒjₛ_allmodes,jₛ)

			jₒjₛ_ind = modeindex(jₒjₛ_allmodes,(jₒ,jₛ))

			Yjₒjₛ_lm_n₁n₂ = Y_lm_jrange_n1n2[jₒjₛ_ind]
			Yjₒjₛ_lm_n₂n₁ = Y_lm_jrange_n2n1[jₒjₛ_ind]

			Gobs1 = Gobs1_cache[jₒ]
			@timeit localtimer "FITS" begin
			# Green functions based at the observation point
			if ω_ind != ω_ind_prev || jₒ ∉ l2_range(jₒjₛ_allmodes,jₛ-1)
				read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
				(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs1,:,1:2,srcindFITS(los),1)
			end
			end # timer
			
			@timeit localtimer "radial term 1" begin
			Gⱼₒⱼₛω_u⁰⁺_firstborn!(Gparts_r₁,Gsrc,drGsrc,jₛ,Gobs1,jₒ,los)
			end # timer

			if !obs_at_same_height

				Gobs2 = Gobs2_cache[jₒ]
				@timeit localtimer "FITS" begin
				if ω_ind != ω_ind_prev || jₒ ∉ l2_range(jₒjₛ_allmodes,jₛ-1)
					read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
					(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs2,:,1:2,srcindFITS(los),1)
				end
				end # timer 

				@timeit localtimer "radial term 1" begin
				Gⱼₒⱼₛω_u⁰⁺_firstborn!(Gparts_r₂,Gsrc,drGsrc,jₛ,Gobs2,jₒ,los)
				end #timer
			end

			l_prev = -1 # something unrealistic to start off
			# Change shmodes to ML to avoid recomputing the radial term for same l

			Dlmn = first(Dlmn_arr)

			modesML = intersect(ML(firstshmodes(Yjₒjₛ_lm_n₁n₂)),ML(SHModes))

			isnothing(modesML) && continue

			for (l,m) in modesML
				# Check if triangle condition is satisfied
				# The outer loop is over all possible jₛ and jₒ for a given l_max
				# Not all combinations would contribute towards a specific l
				δ(jₛ,jₒ,l) || continue

				if l != l_prev
					@timeit localtimer "radial term 2" begin
					Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(Hγℓjₒjₛ_r₁r₂,Gparts_r₁,Gγℓjₒjₛ_r₁,Gα₂r_r₂_rₛ,jₒ,jₛ,l)

					if !obs_at_same_height
						Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(Hγℓjₒjₛ_r₂r₁,Gparts_r₂,Gγℓjₒjₛ_r₂,Gα₂r_r₁_rₛ,jₒ,jₛ,l)
					end

					@. twoimagconjhωconjHγℓjₒjₛ_r₂r₁ = -2imag(h_ω * Hγℓjₒjₛ_r₂r₁)
					@. twoimagconjhωHγℓjₒjₛ_r₁r₂ = 2imag(conjhω * Hγℓjₒjₛ_r₁r₂)

					@inbounds C⁰ = C[0,l,jₒ,jₛ]
					@inbounds C⁺ = C[1,l,jₒ,jₛ]
					phase = (-1)^(jₒ + jₛ + l)
					coeffj = √((2jₒ+1)*(2jₛ+1)/π/(2l+1))

					pre⁰ = dωω³Pω * coeffj * C⁰
					pre⁺ = dωω³Pω * coeffj * Ωjₛ0 * C⁺

					Dlmn = Dlmn_arr[l]

					l_prev = l

					end #timer
				end

				@timeit localtimer "kernel" begin
				populatekernel!(flows(),los,K,
					SHModes,
					Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
					(l,m),(jₛ,jₒ,ω),
					twoimagconjhωHγℓjₒjₛ_r₁r₂,
					twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
					l1,l2,pre⁰,pre⁺,phase,temp,
					Dlmn)
				end #timer
			end
		end

		ω_ind_prev = ω_ind

		signaltomaster!(progress_channel)
	end

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2))

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	mulprefactor!(K)

	return parent(K)
end

# Compute Kₛₜ first and then compute Kₛ₀_rθϕ from that
function kernel_uₛ₀_rθϕ_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_direction,SHModes,hω_arr,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=r_obs_default,
	eulerangles = nothing,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	K = kernel_uₛₜ_partial(ℓ_ωind_iter_on_proc,
		xobs1,xobs2,los,SHModes,hω_arr,
		p_Gsrc,p_Gobs1,p_Gobs2,r_src,r_obs,0:1,
		eulerangles,
		progress_channel,timers_channel)

	K_rθϕ = zeros(size(K,1),3,size(K,3))

	zeroind = axes(K,1)[2]
	plusind = axes(K,1)[3]

	@inbounds for st_ind in axes(K,3)
		for r_ind in axes(K,1)
			# r-component
			K_rθϕ[r_ind,1,st_ind] = real(K[r_ind,zeroind,st_ind])
		end
		for r_ind in axes(K,1)
			# θ-component
			K_rθϕ[r_ind,2,st_ind] = 2real(K[r_ind,plusind,st_ind])
		end
		for r_ind in axes(K,1)
			# ϕ-component
			K_rθϕ[r_ind,3,st_ind] = -2imag(K[r_ind,plusind,st_ind])
		end
	end

	K_rθϕ
end

# Compute Kₛ₀_rθϕ directly
function kernel_uₛ₀_rθϕ_partial_2(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_direction,SHModes,hω_arr,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=r_obs_default,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	@unpack p_Gsrc,p_Gobs1,p_Gobs2,Gfn_path_src,NGfn_files_src,Gfn_path_obs1,NGfn_files_obs1,
	Gfn_path_obs2,NGfn_files_obs2 = unpackGfnparams(p_Gsrc,r_src,p_Gobs1,xobs1.r,p_Gobs2,xobs2.r)

	@unpack ℓ_arr,ω_arr,Nν_Gfn,dω = p_Gsrc

	s_max = l_max(SHModes)

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))
	obs_at_same_height = r₁_ind == r₂_ind

	ℓ_range_proc = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)

	# Get a list of all modes that will be accessed.
	# This can be used to open the fits files before the loops begin.
	# This will cut down on FITS IO costs
	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	# Gℓ′ω(r,robs) files
	Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_obs1,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,s_max,NGfn_files_obs1)
	# Gℓ′ω(r,robs) files
	Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_obs2,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,s_max,NGfn_files_obs2)

	K = zeros(nr,3,length(SHModes)) # Kγₗₘ(r,x₁,x₂)

	arrs = allocatearrays(flows(),los,obs_at_same_height)
	@unpack Gsrc,drGsrc,Gobs1,Gobs2,Gparts_r₁,Gparts_r₂,
	Gγℓjₒjₛ_r₁,Gγℓjₒjₛ_r₂,Hγℓjₒjₛ_r₁r₂,Hγℓjₒjₛ_r₂r₁,temp = arrs 
	@unpack twoimagconjhωconjHγℓjₒjₛ_r₂r₁,twoimagconjhωHγℓjₒjₛ_r₁r₂ = arrs

	jₒjₛ_allmodes = L2L1Triangle(ℓ_range_proc,s_max,ℓ_arr)

	@timeit localtimer "BiPoSH" begin
		Y_lm_jrange_n1n2,Y_lm_jrange_n2n1 = BiPoSH_n1n2_n2n1(
			shtype(los),VSHtype(xobs1,xobs2),xobs1,xobs2,SHModes,jₒjₛ_allmodes)
	end #localtimer

	# Line of sight (if necessary)
	l1,l2 = line_of_sight_covariant(xobs1,xobs2,los)

	C = zeros(0:1,0:s_max,l2_range(jₒjₛ_allmodes),ℓ_range_proc)

	@timeit localtimer "CG" begin
	@inbounds for jₛ in axes(C,4), jₒ in axes(C,3), ℓ in axes(C,2)
		
		C[0,ℓ,jₒ,jₛ] = if iseven(jₒ+jₛ+ℓ)
							clebschgordan(Float64,jₒ,0,jₛ,0,ℓ,0)
						else
							zero(Float64)
						end

		C[1,ℓ,jₒ,jₛ] = if ℓ > 0 && jₛ > 0
							clebschgordan(Float64,jₒ,0,jₛ,1,ℓ,1)
						else
							zero(Float64)
						end
	end
	end # timer

	C⁰,C⁺ = zero(Float64),zero(Float64)
	phase = 1
	pre⁰,pre⁺ = zero(Float64), zero(Float64)

	ω_ind_prev = first(ℓ_ωind_iter_on_proc)[2] - 1

	# Loop over the Greenfn files
	for (jₛ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]
		conjhω = conj(hω_arr[ω_ind])
		dωω³Pω = dω/2π * ω^3 * Powspec(ω)
		Ωjₛ0 = Ω(jₛ,0)

		@timeit localtimer "FITS" begin
		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
		(ℓ_arr,1:Nν_Gfn),(jₛ,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
		(ℓ_arr,1:Nν_Gfn),(jₛ,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),2)
		end # timer

		Gα₂r_r₁_rₛ = αrcomp(Gsrc,r₁_ind,los)
		Gα₂r_r₂_rₛ = αrcomp(Gsrc,r₂_ind,los)

		for jₒ in l2_range(jₒjₛ_allmodes,jₛ)

			jₒjₛ_ind = modeindex(jₒjₛ_allmodes,(jₒ,jₛ))

			Yjₒjₛ_lm_n₁n₂ = Y_lm_jrange_n1n2[jₒjₛ_ind]
			Yjₒjₛ_lm_n₂n₁ = Y_lm_jrange_n2n1[jₒjₛ_ind]

			@timeit localtimer "FITS" begin
			# Green functions based at the observation point
			read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs1,:,1:2,srcindFITS(los),1)
			end # timer
			
			@timeit localtimer "radial term 1" begin
			Gⱼₒⱼₛω_u⁰⁺_firstborn!(Gparts_r₁,Gsrc,drGsrc,jₛ,Gobs1,jₒ,los)
			end # timer

			if !obs_at_same_height
				@timeit localtimer "FITS" begin
					read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
					(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs2,:,1:2,srcindFITS(los),1)
				end # timer 

				@timeit localtimer "radial term 1" begin
				Gⱼₒⱼₛω_u⁰⁺_firstborn!(Gparts_r₂,Gsrc,drGsrc,jₛ,Gobs2,jₒ,los)
				end #timer
			end

			for (l,m) in firstshmodes(Yjₒjₛ_lm_n₁n₂)
				# Check if triangle condition is satisfied
				# The outer loop is over all possible jₛ and jₒ for a given s_max
				# Not all combinations would contribute towards a specific s
				δ(jₛ,jₒ,l) || continue

				@timeit localtimer "radial term 2" begin
				Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(Hγℓjₒjₛ_r₁r₂,Gparts_r₁,Gγℓjₒjₛ_r₁,Gα₂r_r₂_rₛ,jₒ,jₛ,l)

				if !obs_at_same_height
					Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(Hγℓjₒjₛ_r₂r₁,Gparts_r₂,Gγℓjₒjₛ_r₂,Gα₂r_r₁_rₛ,jₒ,jₛ,l)
				end
				end #timer

				@. twoimagconjhωconjHγℓjₒjₛ_r₂r₁ = 2imag(conjhω * conj(Hγℓjₒjₛ_r₂r₁) )
				@. twoimagconjhωHγℓjₒjₛ_r₁r₂ = 2imag(conjhω * Hγℓjₒjₛ_r₁r₂)

				@inbounds C⁰ = C[0,l,jₒ,jₛ]
				@inbounds C⁺ = C[1,l,jₒ,jₛ]
				phase = (-1)^(jₒ + jₛ + l)
				coeffj = √((2jₒ+1)*(2jₛ+1)/π/(2l+1))

				pre⁰ = dωω³Pω * coeffj * C⁰
				pre⁺ = dωω³Pω * coeffj * Ωjₛ0 * C⁺

				@timeit localtimer "kernel" begin
				populatekernelrθϕl0!(flows(),los,K,SHModes,Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
				(l,m),(jₛ,jₒ,ω),conjhω,Hγℓjₒjₛ_r₁r₂,Hγℓjₒjₛ_r₂r₁,
				l1,l2,pre⁰,pre⁺,phase,temp)
				end #timer
			end
		end

		signaltomaster!(progress_channel)
	end

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2))

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	@inbounds for lmind in axes(K,3), vind in axes(K,2), r_ind in axes(K,1)
		K[r_ind,vind,lmind] *= ρ[r_ind]
	end
	
	@inbounds for lmind in axes(K,3), vind in axes(K,2)
		vind == 1 && continue
		for r_ind in axes(K,1)
			K[r_ind,vind,lmind] /= r[r_ind]
		end
	end
	
	return K
end

function _K_uₛₜ(m::SeismicMeasurement,xobs1,xobs2,los::los_direction,
	args...;kwargs...)

	r_src,r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc,p_Gobs1,p_Gobs2 = read_parameters_for_points(xobs1,xobs2;kwargs...)
	@unpack ν_arr,ℓ_arr,ν_start_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	eulerangles = get(kwargs, :eulerangles, nothing)

	hω_arr = get(kwargs,:hω) do
		hω(m,xobs1,xobs2,los;kwargs...,
			eulerangles = eulerangles,
			print_timings=false)
	end

	K_components = get(kwargs,:K_components,-1:1)

	println("Computing kernel")
	K_uₛₜ = pmapsum_timed(kernel_uₛₜ_partial,modes_iter,
		xobs1,xobs2,los,args...,hω_arr,p_Gsrc,p_Gobs1,p_Gobs2,
		r_src,r_obs,K_components,eulerangles;
		progress_str="Modes summed in kernel : ",
		print_timings=get(kwargs,:print_timings,false))

	return ℓ_range,ν_ind_range,K_uₛₜ
end

function _K_uₛₜ_noreduce(m::SeismicMeasurement,xobs1,xobs2,los::los_direction,
	args...;kwargs...)

	r_src,r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc,p_Gobs1,p_Gobs2 = read_parameters_for_points(xobs1,xobs2;kwargs...)
	@unpack ν_arr,ℓ_arr,ν_start_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter,np = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	hω_arr = get(kwargs,:hω) do
		hω(m,xobs1,xobs2,los;kwargs...,print_timings=false)
	end

	K_components = get(kwargs,:K_components,-1:1)

	println("Computing kernel")
	pmapbatch_timed(kernel_uₛₜ_partial,modes_iter,
		xobs1,xobs2,los,args...,hω_arr,p_Gsrc,p_Gobs1,p_Gobs2,
		r_src,r_obs,K_components;
		progress_str="Modes summed in kernel : ",
		print_timings=get(kwargs,:print_timings,false))
	return ℓ_range,ν_ind_range,np
end

# Compute Kₛₜ first and then compute Kₛ₀_rθϕ from that
function _K_uₛ₀_rθϕ(m::SeismicMeasurement,xobs1,xobs2,los::los_direction,args...;kwargs...)
	r_src,r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc,p_Gobs1,p_Gobs2 = read_parameters_for_points(xobs1,xobs2;kwargs...)
	@unpack ν_arr,ℓ_arr,ν_start_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	eulerangles = get(kwargs,:eulerangles,nothing)

	hω_arr = get(kwargs,:hω) do
		hω(m,xobs1,xobs2,los;kwargs...,
			eulerangles = eulerangles,
			print_timings=false)
	end

	println("Computing kernel")
	K_uₛ₀ = pmapsum_timed(kernel_uₛ₀_rθϕ_partial,modes_iter,
		xobs1,xobs2,los,args...,hω_arr,p_Gsrc,p_Gobs1,p_Gobs2,
		r_src,r_obs,eulerangles;
		progress_str="Modes summed in kernel : ",
		print_timings=get(kwargs,:print_timings,false))

	return ℓ_range,ν_ind_range,K_uₛ₀
end

# Compute Kₛ₀_rθϕ directly
function _K_uₛ₀_rθϕ_2(m::SeismicMeasurement,xobs1,xobs2,los::los_direction,args...;kwargs...)
	r_src,r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc,p_Gobs1,p_Gobs2 = read_parameters_for_points(xobs1,xobs2;kwargs...)
	@unpack ν_arr,ℓ_arr,ν_start_zeros = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	hω_arr = get(kwargs,:hω) do
		hω(m,xobs1,xobs2,los;kwargs...,print_timings=false)
	end

	println("Computing kernel")
	K_uₛ₀ = pmapsum_timed(kernel_uₛ₀_rθϕ_partial_2,modes_iter,
		xobs1,xobs2,los,args...,hω_arr,p_Gsrc,p_Gobs1,p_Gobs2,
		r_src,r_obs;
		progress_str="Modes summed in kernel : ",
		print_timings=get(kwargs,:print_timings,false))

	return ℓ_range,ν_ind_range,K_uₛ₀
end

@def generatefitsheader begin
	header = FITSHeader(["r1","th1","phi1","r2","th2","phi2",
		"l_min","m_min","l_max","m_max",
		"j_min","j_max","nui_min","nui_max"],
		Any[float(xobs1.r),float(xobs1.θ),float(xobs1.ϕ),
		float(xobs2.r),float(xobs2.θ),float(xobs2.ϕ),
		l_min(SHModes), m_min(SHModes), l_max(SHModes), m_max(SHModes),
		minimum(j_range),maximum(j_range),
		minimum(ν_ind_range),maximum(ν_ind_range)],
		["Radius of the first observation point",
		"Colatitude of the first observation point",
		"Azimuth of the first observation point",
		"Radius of the second observation point",
		"Colatitude of the second observation point",
		"Azimuth of the second observation point",
		"Minimum angular degree of the flow",
		"Minimum azimuthal order of the flow",
		"Maximum angular degree of the flow",
		"Maximum azimuthal order of the flow",
		"Minimum wave mode","Maximum wave mode",
		"Minimum wave frequency index","Maximum wave frequency index"])
end

function modetag(j_range,SHModes)
   "jmax$(maximum(j_range))_lmax$(l_max(SHModes))_mmax$(m_max(SHModes))"
end
function kernelfilenameuₛₜ(::TravelTimes,::los_radial,j_range,SHModes,tag="")
	"Kst_δτ_u_$(modetag(j_range,SHModes))"*string(tag)*".fits"
end
function kernelfilenameuₛₜ(::TravelTimes,::los_earth,j_range,SHModes,tag="")
	"Kst_δτ_u_$(modetag(j_range,SHModes))_los"*string(tag)*".fits"
end
function kernelfilenameuₛₜ(::Amplitudes,::los_radial,j_range,SHModes,tag="")
	"Kst_A_u_$(modetag(j_range,SHModes))"*string(tag)*".fits"
end
function kernelfilenameuₛₜ(::Amplitudes,::los_earth,j_range,SHModes,tag="")
	"Kst_A_u_$(modetag(j_range,SHModes))_los"*string(tag)*".fits"
end

filenamerottag(::Nothing) = ""
filenamerottag(::NTuple{3,Real}) = "_rot"

function kernel_uₛₜ(m::SeismicMeasurement,xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	SHModes = getkernelmodes(;kwargs...)

	j_range,ν_ind_range,K_δτ_uₛₜ = _K_uₛₜ(m,xobs1,xobs2,los,SHModes;kwargs...)
	K = OffsetArray(K_δτ_uₛₜ,axes(K_δτ_uₛₜ)[1],-1:1,axes(K_δτ_uₛₜ)[3])
	
	eulerangles = get(kwargs,:eulerangles,nothing)
	filenametag = get(kwargs,:filenametag) do 
		filenamerottag(eulerangles)
	end

	if get(kwargs,:save,true)
		
		filepath = joinpath(SCRATCH_kerneldir,
			kernelfilenameuₛₜ(m,los,j_range,SHModes,filenametag))

		@generatefitsheader

		FITS(filepath,"w") do f
			write(f,reinterpret_as_float(K_δτ_uₛₜ),header=header)
		end
	end
	
	SHArray(K,(axes(K)[1:2]...,SHModes))
end

@two_points_on_the_surface kernel_uₛₜ

kernelfilenameuₛ₀rθϕ(::TravelTimes,::los_radial) = "Kl0_δτ_u_rθϕ.fits"
kernelfilenameuₛ₀rθϕ(::TravelTimes,::los_earth) = "Kl0_δτ_u_rθϕ_los.fits"
kernelfilenameuₛ₀rθϕ(::Amplitudes,::los_radial) = "Kl0_A_u_rθϕ.fits"
kernelfilenameuₛ₀rθϕ(::Amplitudes,::los_earth) = "Kl0_A_u_rθϕ_los.fits"

# Compute Kₛₜ first and then compute Kₛ₀_rθϕ from that
function kernel_uₛ₀_rθϕ(m::SeismicMeasurement,xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	if isnothing(get(kwargs,:SHModes,nothing))
		s_min = get(kwargs,:s_min,0)
		s_max = get(kwargs,:s_max,s_min)
	else
		modes = kwargs.data.SHModes
		s_min, s_max = l_minmax(modes)
	end

	SHModes = LM(s_min:s_max, SingleValuedRange(0))

	j_range,ν_ind_range,K_δτ_uₛ₀ = _K_uₛ₀_rθϕ(m,xobs1,xobs2,los,SHModes;kwargs...)
	
	if get(kwargs,:save,true)
		filename = joinpath(SCRATCH_kerneldir,kernelfilenameuₛ₀rθϕ(m,los))
		@generatefitsheader

		FITS(filename,"w") do f
			write(f,reinterpret_as_float(K_δτ_uₛ₀),header=header)
		end
	end

	SHArray(K_δτ_uₛ₀,(axes(K_δτ_uₛ₀)[1:2]...,SHModes))
end
@two_points_on_the_surface kernel_uₛ₀_rθϕ

# Compute Kₛ₀_rθϕ directly
function kernel_uₛ₀_rθϕ_2(m::SeismicMeasurement,xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	if isnothing(get(kwargs,:SHModes,nothing))
		s_min = get(kwargs,:s_min,0)
		s_max = get(kwargs,:s_max,s_min)
	else
		modes = kwargs.data.SHModes
		s_min, s_max = l_minmax(modes)
	end

	SHModes = LM(s_min:s_max, SingleValuedRange(0))

	j_range,ν_ind_range,K_δτ_uₛ₀ = _K_uₛ₀_rθϕ_2(m,xobs1,xobs2,los,SHModes;kwargs...)
	
	if get(kwargs,:save,true)
		filename = joinpath(SCRATCH_kerneldir,kernelfilenameuₛ₀rθϕ(m,los))
		@generatefitsheader

		FITS(filename,"w") do f
			write(f,reinterpret_as_float(K_δτ_uₛ₀),header=header)
		end
	end

	SHArray(K_δτ_uₛ₀,(axes(K_δτ_uₛ₀)[1:2]...,SHModes))
end
@two_points_on_the_surface kernel_uₛ₀_rθϕ_2

function kernel_ψϕₛ₀(m::SeismicMeasurement,xobs1,xobs2,los::los_direction=los_radial();kwargs...)

	Kv = kernel_uₛ₀_rθϕ(m,xobs1,xobs2,los;kwargs...,save=false)

	mode_range = firstshmodes(Kv);
	Kψϕ = SHArray{ComplexF64}((axes(Kv,1),mode_range));

	kernel_ψϕₛ₀!(Kψϕ,Kv;save=get(kwargs,:save,true))

	if get(kwargs,:save,true)
		filename = joinpath(SCRATCH_kerneldir,"Kψ_imag.FITS")
		FITS(filename,"w") do f
			write(f,reinterpret_as_float(Kψϕ))
		end
	end

	return Kψϕ
end

function kernel_ψϕₛ₀!(Kψϕ::SHArray,Kv::SHArray;kwargs...)

	mode_range = firstshmodes(Kv)

	for l in l_range(mode_range)
		@views Kψϕ[:,(l,0)] .= ddr*(Kv[:,2,(l,0)]./ρ) .+ 
			@. (Kv[:,2,(l,0)] - 2Ω(l,0)*Kv[:,1,(l,0)])/(ρ*r)
	end

	return Kψϕ
end

#################################################################################################################
# Validation for isotropic sound speed perturbation
#################################################################################################################

function allocatearrays(::soundspeed,los::los_direction,obs_at_same_height)
	Gsrc = zeros_Float64_to_ComplexF64(1:nr,0:1,srcindG(los)...)
	drGsrc = zeros_Float64_to_ComplexF64(1:nr,0:0,srcindG(los)...)
	Gobs1 = zeros_Float64_to_ComplexF64(1:nr,0:1,srcindG(los)...)
	drGobs1 = zeros_Float64_to_ComplexF64(1:nr,0:0,srcindG(los)...)
	Gobs2,drGobs2 = Gobs1,drGobs1

	Gobs1_cache = Dict{Int,typeof(Gobs1)}()
	drGobs1_cache = Dict{Int,typeof(drGobs1)}()
	Gobs2_cache = obs_at_same_height ? Gobs1_cache : Dict{Int,typeof(Gobs2)}()
	drGobs2_cache = obs_at_same_height ? drGobs1_cache : Dict{Int,typeof(drGobs2)}()

	divGsrc = zeros(ComplexF64,nr,srcindG(los)...)
	divGobs = zeros(ComplexF64,nr,srcindG(los)...)
	
	# f_αjₒjₛ(r,rᵢ,rₛ) = -2ρc ∇⋅Gjₒ(r,rᵢ)_α ∇⋅Gjₛ(r,rₛ)_0
	fjₒjₛ_r₁_rsrc = zeros(ComplexF64,nr,srcindG(los)...)
	fjₒjₛ_r₂_rsrc = obs_at_same_height ? fjₒjₛ_r₁_rsrc : zero(fjₒjₛ_r₁_rsrc)

	# H_βαjₒjₛ(r;r₁,r₂,rₛ) = conj(f_αjₒjₛ(r,r₁,rₛ)) Gβ0jₛ(r₂,rₛ)
	Hjₒjₛω_r₁r₂ = zeros(ComplexF64,nr,obsindG(los)...,srcindG(los)...)
	Hjₒjₛω_r₂r₁ = obs_at_same_height ? Hjₒjₛω_r₁r₂ : zero(Hjₒjₛω_r₁r₂)

	tworealconjhωHjₒjₛω_r₁r₂ = zeros(nr,obsindG(los)...,srcindG(los)...)
	tworealconjhωconjHjₒjₛω_r₂r₁ = zero(tworealconjhωHjₒjₛω_r₁r₂)

	@eponymtuple(Gsrc,drGsrc,Gobs1,drGobs1,Gobs2,drGobs2,
		Gobs1_cache,drGobs1_cache,Gobs2_cache,drGobs2_cache,
		divGsrc,divGobs,
		fjₒjₛ_r₁_rsrc,fjₒjₛ_r₂_rsrc,
		Hjₒjₛω_r₁r₂,Hjₒjₛω_r₂r₁,
		tworealconjhωHjₒjₛω_r₁r₂,tworealconjhωconjHjₒjₛω_r₂r₁)
end

function populatekernelvalidation!(::soundspeed,::los_radial,K::AbstractVector{<:Real},
	(ℓ,ω),Y12ℓ,conjhω,H¹₁jjω_r₁r₂,H¹₁jj_r₂r₁,dω,l1,l2)

	pre = 1/√(4π) * dω/2π * ω^2 * Powspec(ω) * (2ℓ+1)/4π * Y12ℓ
	@. K += pre * 2real(conjhω * ( H¹₁jjω_r₁r₂ + conj(H¹₁jj_r₂r₁) ) )
end

function populatekernelvalidation!(::soundspeed,::los_radial,K::AbstractMatrix{<:Real},
	(ℓ,ω),Y12ℓ,conjhω,H¹₁jjω_r₁r₂::AbstractVector{<:Complex},
	H¹₁jj_r₂r₁::AbstractVector{<:Complex},dω,l1,l2arr)

	pre = 1/√(4π) * dω/2π * ω^2 * Powspec(ω) * (2ℓ+1)/4π

	@inbounds for n2ind in axes(K,2)
		temp_ωℓ_n2 = pre * Y12ℓ[n2ind]
		conjhω_n2 = conjhω[n2ind]
		for r_ind in axes(K,1)
			K[r_ind,n2ind] += temp_ωℓ_n2 * 
				2real(conjhω_n2 * ( H¹₁jjω_r₁r₂[r_ind] + conj(H¹₁jj_r₂r₁[r_ind]) ) )
		end
	end
end

function populatekernelvalidation!(::soundspeed,::los_earth,K::AbstractVector{<:Real},
	(j,ω),Y12j,conjhω,Hjj_r₁r₂,Hjj_r₂r₁,dω,l1,l2)

	pre = dω/2π * ω^2 * Powspec(ω) * (-1)^j * √((2j+1)/π)
	ind00 = modeindex(firstshmodes(Y12j),(0,0))

	@inbounds for α₂ in axes(Y12j,2),α₁ in axes(Y12j,1)

		llY = pre * real(l1[α₁] * l2[α₂] * Y12j[α₁,α₂,ind00])
		iszero(llY) && continue

		for r_ind in axes(K,1)
			K[r_ind] += real(conjhω * 
				( Hjj_r₁r₂[r_ind,abs(α₁),abs(α₂)] + 
					conj(Hjj_r₂r₁[r_ind,abs(α₂),abs(α₁)]) ) ) * llY
		end
	end
end

function populatekernelvalidation!(::soundspeed,::los_earth,K::AbstractMatrix{<:Real},
	(j,ω),Y12j,conjhω,Hjjω_r₁r₂,Hjj_r₂r₁,dω,l1,l2arr)

	pre = dω/2π * ω^2 * Powspec(ω) * (-1)^j * √((2j+1)/π)
	ind00 = modeindex(firstshmodes(first(Y12j)),(0,0))

	@inbounds for n2ind in axes(K,2)
		Yjn₂ = Y12j[n2ind]
		ln₂ = l2arr[n2ind]
		conjhω_n2 = conjhω[n2ind]

		for α₂ in axes(Yjn₂,2),α₁ in axes(Yjn₂,1)

			llY = pre * real(l1[α₁] * ln₂[α₂] * Yjn₂[α₁,α₂,ind00])
			iszero(llY) && continue

			for r_ind in axes(K,1)
				K[r_ind,n2ind] += real(conjhω_n2 * 
					( Hjjω_r₁r₂[r_ind,abs(α₁),abs(α₂)] + conj(Hjj_r₂r₁[r_ind,abs(α₂),abs(α₁)]) ) ) * llY
			end
		end
	end
end

function kernel_δc₀₀_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_direction,hω,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	@unpack p_Gsrc,p_Gobs1,p_Gobs2,Gfn_path_src,NGfn_files_src,
	Gfn_path_obs1,NGfn_files_obs1,Gfn_path_obs2,NGfn_files_obs2 = 
		unpackGfnparams(p_Gsrc,r_src,p_Gobs1,xobs1.r,p_Gobs2,xobs2.r)

	@unpack ℓ_arr,ω_arr,Nν_Gfn,dω = p_Gsrc

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))

	Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2 = 
		Gfn_fits_files((Gfn_path_src,Gfn_path_obs1,Gfn_path_obs2),
			(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,
			(NGfn_files_src,NGfn_files_obs1,NGfn_files_obs2))

	K = zeros(nr)

	arr = allocatearrays(soundspeed(),los,r₁_ind == r₂_ind)
	@unpack Gsrc,drGsrc,divGsrc,Gobs1,drGobs1,Gobs2,divGobs = arr
	fjj_r₁_rsrc,fjj_r₂_rsrc = arr.fjₒjₛ_r₁_rsrc, arr.fjₒjₛ_r₂_rsrc
	H¹₁jj_r₁r₂,H¹₁jj_r₂r₁ = arr.Hjₒjₛω_r₁r₂, arr.Hjₒjₛω_r₂r₁

	ℓ_range = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	Y12 = computeY₀₀(los,xobs1,xobs2,ℓ_range)

	# covariant components
	l1,l2 = line_of_sight_covariant(xobs1,xobs2,los)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]
		conjhω = conj(hω[ω_ind])

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:1,srcindFITS(los),2)

		Gγr_r₁_rsrc = αrcomp(Gsrc,r₁_ind,los)
		Gγr_r₂_rsrc = αrcomp(Gsrc,r₂_ind,los)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs1,:,1:2,srcindFITS(los),1)

		# Derivative of Green function about receiver location
		read_Gfn_file_at_index!(drGobs1,Gfn_fits_files_obs1,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs1,:,1:1,srcindFITS(los),2)

		radial_fn_isotropic_δc_firstborn!(fjj_r₁_rsrc,
			Gsrc,drGsrc,divGsrc,Gobs1,drGobs1,divGobs,ℓ)

		Hjₒjₛω!(H¹₁jj_r₁r₂,fjj_r₁_rsrc,Gγr_r₂_rsrc)

		if r₁_ind != r₂_ind
			# Green function about receiver location
			read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
				(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs2,:,1:2,srcindFITS(los),1)

			# Derivative of Green function about receiver location
			read_Gfn_file_at_index!(drGobs2,Gfn_fits_files_obs2,
				(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs2,:,1:1,srcindFITS(los),2)

			radial_fn_isotropic_δc_firstborn!(fjj_r₂_rsrc,
				Gsrc,drGsrc,divGsrc,Gobs2,drGobs2,divGobs,ℓ)

			Hjₒjₛω!(H¹₁jj_r₂r₁,fjj_r₂_rsrc,Gγr_r₁_rsrc)
		end

		populatekernelvalidation!(soundspeed(),los,K,
		(ℓ,ω),Y12[ℓ],conjhω,H¹₁jj_r₁r₂,H¹₁jj_r₂r₁,dω,l1,l2)

		signaltomaster!(progress_channel)
	end

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2))

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	return K
end

function kernel_δc₀₀_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	nobs1::Point2D,nobs2_arr::Vector{<:Point2D},los::los_direction,hω_arr,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=p_Gobs1,
	r_src=r_src_default,r_obs=r_obs_default,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	hω_arr = permutedims(hω_arr) #  convert to (n2,ω)

	r_obs_ind = radial_grid_index(r_obs)

	@unpack p_Gsrc,p_Gobs1,Gfn_path_src,NGfn_files_src = 
		unpackGfnparams(p_Gsrc,r_src,p_Gobs1,r_obs,p_Gobs2,r_obs)

	Gfn_path_obs,NGfn_files_obs = p_Gobs1.path,p_Gobs1.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn,dω = p_Gsrc

	Gfn_fits_files_src,Gfn_fits_files_obs = 
		Gfn_fits_files((Gfn_path_src,Gfn_path_obs),
			(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,
			(NGfn_files_src,NGfn_files_obs))

	K = zeros(nr,length(nobs2_arr))

	arrs = allocatearrays(soundspeed(),los,true)
	@unpack Gsrc,drGsrc,divGsrc,divGobs = arrs
	Gobs, drGobs = arrs.Gobs1, arrs.drGobs1
	f_robs_rsrc, Hjj_robs_rsrc = arrs.fjₒjₛ_r₁_rsrc, arrs.Hjₒjₛω_r₁r₂

	ℓ_range = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	Y12 = computeY₀₀(los,nobs1,nobs2_arr,ℓ_range)

	# covariant components
	l1,l2arr = line_of_sight_covariant(nobs1,nobs2_arr,los)

	for (ℓ,ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]
		conjhω = conj(@view hω_arr[:,ω_ind])

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_src,:,1:1,srcindFITS(los),2)

		Gγr_robs_rsrc = αrcomp(Gsrc,r_obs_ind,los)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs,Gfn_fits_files_obs,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs,:,1:2,srcindFITS(los),1)

		# Derivative of Green function about receiver location
		read_Gfn_file_at_index!(drGobs,Gfn_fits_files_obs,
			(ℓ_arr,1:Nν_Gfn),(ℓ,ω_ind),NGfn_files_obs,:,1:1,srcindFITS(los),2)

		radial_fn_isotropic_δc_firstborn!(f_robs_rsrc,
			Gsrc,drGsrc,divGsrc,Gobs,drGobs,divGobs,ℓ)

		Hjₒjₛω!(Hjj_robs_rsrc,f_robs_rsrc,Gγr_robs_rsrc)

		populatekernelvalidation!(soundspeed(),los,K,
		(ℓ,ω),Y12[ℓ],conjhω,Hjj_robs_rsrc,Hjj_robs_rsrc,dω,l1,l2arr)

		signaltomaster!(progress_channel)
	end

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs))

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	return K
end

function _K_δc₀₀(m::SeismicMeasurement,xobs1,xobs2,los::los_direction;kwargs...)
	r_src,r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc,p_Gobs1,p_Gobs2 = read_parameters_for_points(xobs1,xobs2;kwargs...,c_scale=1)
	@unpack ν_arr,ℓ_arr = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	hω_arr = get(kwargs,:hω) do
		hω(m,xobs1,xobs2,los;kwargs...,print_timings=false)
	end

	K_δc₀₀ = pmapsum_timed(kernel_δc₀₀_partial,modes_iter,
		xobs1,xobs2,los,hω_arr,p_Gsrc,p_Gobs1,p_Gobs2,r_src,r_obs;
		progress_str="Modes summed in kernel : ",
		print_timings=get(kwargs,:print_timings,false))
end

kernelfilenameδc₀₀(::TravelTimes,::los_radial) = "K_δτ_δc₀₀.fits"
kernelfilenameδc₀₀(::TravelTimes,::los_earth) = "K_δτ_δc₀₀_los.fits"
kernelfilenameδc₀₀(::Amplitudes,::los_radial) = "K_A_δc₀₀.fits"
kernelfilenameδc₀₀(::Amplitudes,::los_earth) = "K_A_δc₀₀_los.fits"

function kernel_δc₀₀(m::SeismicMeasurement,xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	K = _K_δc₀₀(m,xobs1,xobs2,los;kwargs...)
	
	if get(kwargs,:save,true)
		filename = kernelfilenameδc₀₀(m,los)
		FITS(filename,"w") do f
			write(f,reinterpret_as_float(K))
		end
	end

	return K
end

@two_points_on_the_surface kernel_δc₀₀

########################################################################################
# Sound-speed kernels
########################################################################################

function populatekernel!(::soundspeed,::los_radial,K::AbstractMatrix{ComplexF64},
	SHModes,
	Yjₒjₛ_n1n2::SHArray{<:Complex,1},
	Yjₒjₛ_n2n1::SHArray{<:Complex,1},
	(l,m)::NTuple{2,Integer},
	(jₛ,jₒ,ω)::Tuple{Integer,Integer,Real},
	pre,tworealconjhωHjₒjₛω_r₁r₂::AbstractVector{<:Real},
	tworealconjhωconjHjₒjₛω_r₂r₁::AbstractVector{<:Real},l1,l2)
	
	lm_ind = modeindex(firstshmodes(Yjₒjₛ_n1n2),(l,m))
	lm_ind_K = modeindex(SHModes,(l,m))

	llY_n₁n₂ = conj(Yjₒjₛ_n1n2[lm_ind])
	llY_n₂n₁ = conj(Yjₒjₛ_n2n1[lm_ind])

	iszero(llY_n₁n₂) && iszero(llY_n₂n₁) && return

	@inbounds for r_ind in axes(K,1)
		K[r_ind,lm_ind_K] +=  pre * (

		tworealconjhωHjₒjₛω_r₁r₂[r_ind] * llY_n₁n₂ + 
		tworealconjhωconjHjₒjₛω_r₂r₁[r_ind] * llY_n₂n₁

		)
	end
end

function populatekernel!(::soundspeed,::los_earth,K::AbstractMatrix{ComplexF64},
	SHModes,
	Yjₒjₛ_lm_n₁n₂::SHArray{<:Complex,3},
	Yjₒjₛ_lm_n₂n₁::SHArray{<:Complex,3},
	(l,m)::NTuple{2,Integer},
	(jₛ,jₒ,ω)::Tuple{Integer,Integer,Real},
	pre,tworealconjhωHjₒjₛω_r₁r₂::AbstractArray{<:Real,3},
	tworealconjhωconjHjₒjₛω_r₂r₁::AbstractArray{<:Real,3},
	ln₁::AbstractVector,ln₂::AbstractVector)
	
	lm_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂),(l,m))
	lm_ind_K = modeindex(SHModes,(l,m))

	@inbounds for α₂ = axes(Yjₒjₛ_lm_n₁n₂,2), α₁ = axes(Yjₒjₛ_lm_n₁n₂,1)
		llY_n₁n₂ = conj(ln₁[α₁]*ln₂[α₂]*Yjₒjₛ_lm_n₁n₂[α₁,α₂,lm_ind])
		llY_n₂n₁ = conj(ln₂[α₁]*ln₁[α₂]*Yjₒjₛ_lm_n₂n₁[α₁,α₂,lm_ind])

		iszero(llY_n₁n₂) && iszero(llY_n₂n₁) && continue

		for r_ind in axes(K,1)
			K[r_ind,lm_ind_K] +=  pre * (

			tworealconjhωHjₒjₛω_r₁r₂[r_ind,abs(α₁),abs(α₂)] * llY_n₁n₂ + 
			tworealconjhωconjHjₒjₛω_r₂r₁[r_ind,abs(α₁),abs(α₂)] * llY_n₂n₁

			)
		end
	end
end

function kernel_δcₛₜ_partial(ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D,xobs2::Point3D,los::los_direction,SHModes,hω_arr,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=nothing,
	r_src=r_src_default,r_obs=nothing,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	@unpack p_Gsrc,p_Gobs1,p_Gobs2,Gfn_path_src,NGfn_files_src,
	Gfn_path_obs1,NGfn_files_obs1,Gfn_path_obs2,NGfn_files_obs2 = 
		unpackGfnparams(p_Gsrc,r_src,p_Gobs1,xobs1.r,p_Gobs2,xobs2.r)

	@unpack ℓ_arr,ω_arr,Nν_Gfn,dω = p_Gsrc

	s_max,t_max = l_max(SHModes), m_max(SHModes)

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))
	obs_at_same_height = r₁_ind == r₂_ind

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,NGfn_files_src)

	# Gℓ′ω(r,robs) files
	Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_obs1,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,s_max,NGfn_files_obs1)

	Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_obs2,
						(ℓ_arr,1:Nν_Gfn),ℓ_ωind_iter_on_proc,s_max,NGfn_files_obs2)

	arrs = allocatearrays(soundspeed(),los,obs_at_same_height)
	@unpack Gsrc,drGsrc,Gobs1,drGobs1,Gobs2,drGobs2,divGsrc,divGobs = arrs
	@unpack fjₒjₛ_r₁_rsrc,fjₒjₛ_r₂_rsrc,Hjₒjₛω_r₁r₂,Hjₒjₛω_r₂r₁ = arrs
	@unpack Gobs1_cache,drGobs1_cache,Gobs2_cache,drGobs2_cache = arrs
	@unpack tworealconjhωHjₒjₛω_r₁r₂,tworealconjhωconjHjₒjₛω_r₂r₁ = arrs

	ℓ_range_proc = UnitRange(extrema(ℓ_ωind_iter_on_proc,dim=1)...)
	
	jₒjₛ_allmodes = L2L1Triangle(ℓ_range_proc,s_max,ℓ_arr)
	jₒrange = l2_range(jₒjₛ_allmodes)
	@unpack Gobs1_cache,Gobs2_cache = arrs
	for jₒ in jₒrange
		Gobs1_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(Gsrc)...)
		drGobs1_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(drGsrc)...)
		if !obs_at_same_height
			Gobs2_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(Gsrc)...)
			drGobs2_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(drGsrc)...)
		end
	end

	@timeit localtimer "CG" begin
		C = SHVector([zeros(abs(jₒ-jₛ):jₒ+jₛ) for (jₒ,jₛ) in jₒjₛ_allmodes],jₒjₛ_allmodes)
		for (jₒ,jₛ) in jₒjₛ_allmodes
			C_jₒjₛ = C[(jₒ,jₛ)]
			for l in abs(jₒ-jₛ):jₒ+jₛ
				if isodd(jₒ+jₛ+l)
					continue
				end
				C_jₒjₛ[l] = clebschgordan(Float64,jₒ,0,jₛ,0,l,0)
			end
		end
	end

	K = zeros(ComplexF64,nr,length(SHModes))

	@timeit localtimer "BiPoSH" begin
	Y_lm_jrange_n1n2,Y_lm_jrange_n2n1 = BiPoSH_n1n2_n2n1(
		shtype(los),VSHtype(xobs1,xobs2),xobs1,xobs2,SHModes,jₒjₛ_allmodes)
	end # timer

	# Line of sight direction (if necessary)
	l1,l2 = line_of_sight_covariant(xobs1,xobs2,los)
	ω_ind_prev = -1
	# Loop over the Greenfn files
	for (jₛ,ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]
		conjhω = conj(hω_arr[ω_ind])
		dωω²Pω = dω/2π * ω^2 * Powspec(ω)

		# memfreeGB = Sys.free_memory()/2^30
		# @show (jₛ,ω_ind,memfreeGB)
		
		@timeit localtimer "FITS" begin

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(jₛ,ω_ind),NGfn_files_src,:,1:2,srcindFITS(los),1)

		Gαr_r₁_rsrc = αrcomp(Gsrc,r₁_ind,los)
		Gαr_r₂_rsrc = αrcomp(Gsrc,r₂_ind,los)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(jₛ,ω_ind),NGfn_files_src,:,1:1,srcindFITS(los),2)
		end #localtimer

		for jₒ in l2_range(jₒjₛ_allmodes,jₛ)

			jₒjₛ_ind = modeindex(jₒjₛ_allmodes,(jₒ,jₛ))

			Yjₒjₛ_lm_n₁n₂ = Y_lm_jrange_n1n2[jₒjₛ_ind]
			Yjₒjₛ_lm_n₂n₁ = Y_lm_jrange_n2n1[jₒjₛ_ind]

			Cljₒjₛ = C[(jₒ,jₛ)]

			Gobs1 = Gobs1_cache[jₒ]
			drGobs1 = drGobs1_cache[jₒ]
			Gobs2 = Gobs2_cache[jₒ]
			drGobs2 = drGobs2_cache[jₒ]

			@timeit localtimer "FITS" begin

			if ω_ind != ω_ind_prev || jₒ ∉ l2_range(jₒjₛ_allmodes,jₛ-1)
				# Green functions based at the observation point for ℓ′
				read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
				(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs1,:,1:2,srcindFITS(los),1)

				read_Gfn_file_at_index!(drGobs1,Gfn_fits_files_obs1,
				(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs1,:,1:1,srcindFITS(los),2)
			end

			end # timer

			@timeit localtimer "radial term" begin

			# precompute the radial term in f
			radial_fn_δc_firstborn!(fjₒjₛ_r₁_rsrc,Gsrc,drGsrc,jₛ,divGsrc,
				Gobs1,drGobs1,jₒ,divGobs)

			end # timer

			@timeit localtimer "radial term 2" begin
				Hjₒjₛω!(Hjₒjₛω_r₁r₂,fjₒjₛ_r₁_rsrc,Gαr_r₂_rsrc)
			end
			@timeit localtimer "radial term 3" begin
				@. tworealconjhωHjₒjₛω_r₁r₂ = 2real(conjhω * Hjₒjₛω_r₁r₂)
			end

			if !obs_at_same_height

				@timeit localtimer "FITS" begin
				if ω_ind != ω_ind_prev || jₒ ∉ l2_range(jₒjₛ_allmodes,jₛ-1)
					read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
					(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs2,:,1:2,srcindFITS(los),1)

					read_Gfn_file_at_index!(drGobs2,Gfn_fits_files_obs2,
					(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs2,:,1:1,srcindFITS(los),2)
				end
				end # timer

				@timeit localtimer "radial term" begin
				radial_fn_δc_firstborn!(fjₒjₛ_r₂_rsrc,Gsrc,drGsrc,jₛ,divGsrc,
					Gobs2,drGobs2,jₒ,divGobs)
				end # timer
				
				@timeit localtimer "radial term 2" begin
				Hjₒjₛω!(Hjₒjₛω_r₂r₁,fjₒjₛ_r₂_rsrc,Gαr_r₁_rsrc)
				end
			end
			@timeit localtimer "radial term 3" begin
			@. tworealconjhωconjHjₒjₛω_r₂r₁ = 2real(conjhω * conj(Hjₒjₛω_r₂r₁))
			end

			# @inbounds for (lm_ind,(l,m)) in enumerate(shmodes(Yjₒjₛ_lm_n₁n₂))
			for (l,m) in firstshmodes(Yjₒjₛ_lm_n₁n₂)

				# The Clebsch-Gordan coefficients imply the selection
				# rule that only even ℓ+ℓ′+s modes contribute
				isodd(jₒ + jₛ + l) && continue

				# Check if triangle condition is satisfied
				# The loop is over all possible ℓ and ℓ′ for a given s_max
				# Not all combinations would contribute towards a specific s
				δ(jₛ,jₒ,l) || continue

				pre = dωω²Pω * Njₒjₛs(jₒ,jₛ,l) * Cljₒjₛ[l]
				
				@timeit localtimer "kernel" begin

				populatekernel!(soundspeed(),los,K,SHModes,
					Yjₒjₛ_lm_n₁n₂,Yjₒjₛ_lm_n₂n₁,
					(l,m),(jₛ,jₒ,ω),pre,
					tworealconjhωHjₒjₛω_r₁r₂,
					tworealconjhωconjHjₒjₛω_r₂r₁,l1,l2)
				end
			end # lm
		end # jₒ

		ω_ind_prev = ω_ind

		signaltomaster!(progress_channel)
	end # (jₛ,ω)

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2))
	
	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))
	
	K
end

function _K_δcₛₜ(m::SeismicMeasurement,xobs1,xobs2,los::los_direction,args...;kwargs...)
	r_src,r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc,p_Gobs1,p_Gobs2 = read_parameters_for_points(xobs1,xobs2;kwargs...,c_scale=1)
	@unpack ν_arr,ℓ_arr = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	hω_arr = get(kwargs,:hω) do
		hω(m,xobs1,xobs2,los;kwargs...,print_timings=false)
	end

	println("Computing kernel")
	K_δcₛₜ = pmapsum_timed(kernel_δcₛₜ_partial,modes_iter,
		xobs1,xobs2,los,args...,hω_arr,p_Gsrc,p_Gobs1,p_Gobs2,r_src,r_obs;
		progress_str="Modes summed in kernel : ",
		print_timings=get(kwargs,:print_timings,false))

	return ℓ_range,ν_ind_range,K_δcₛₜ
end

function kernelfilenameδcₛₜ(::TravelTimes,::los_radial,j_range,SHModes)
	   "Kst_δτ_c_$(modetag(j_range,SHModes)).fits"
end
function kernelfilenameδcₛₜ(::TravelTimes,::los_earth,j_range,SHModes)
	   "Kst_δτ_c_$(modetag(j_range,SHModes))_los.fits"
end
function kernelfilenameδcₛₜ(::Amplitudes,::los_radial,j_range,SHModes)
	   "Kst_A_c_$(modetag(j_range,SHModes)).fits"
end
function kernelfilenameδcₛₜ(::Amplitudes,::los_earth,j_range,SHModes)
	   "Kst_A_c_$(modetag(j_range,SHModes))_los.fits"
end

function kernel_δcₛₜ(m::SeismicMeasurement,xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)
	
	SHModes = getkernelmodes(;kwargs...)
	
	j_range,ν_ind_range,K_δcₛₜ = _K_δcₛₜ(m,xobs1,xobs2,los,SHModes;kwargs...)
	
	if get(kwargs,:save,true)
		filename = joinpath(SCRATCH_kerneldir,kernelfilenameδcₛₜ(m,los,j_range,SHModes))
		@generatefitsheader
		FITS(filename,"w") do f
			write(f,reinterpret_as_float(K_δcₛₜ),header=header)
		end
	end

	SHArray(K_δcₛₜ,(axes(K_δcₛₜ,1),SHModes))
end

@two_points_on_the_surface kernel_δcₛₜ

end # module