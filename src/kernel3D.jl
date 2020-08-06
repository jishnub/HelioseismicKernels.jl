include("$(@__DIR__)/kernel.jl")

module Kernel3D

using ..Kernel
using DistributedArrays
using SphericalHarmonics
using FastGaussQuadrature
using WignerD
using SphericalHarmonicModes
import WignerD: Jy_eigen!, GeneralizedY, ClampedWignerdMatrix, vectorinds

function populatekernel!(::flows,K,Kₗₘ,Y,modes_lm,l,K_all_modes,Ω_ind,temp⁰,temp⁺¹)

	# The kernel is defined as 
	# Kr = Σ_l K₀,ₗ₀*(r) Yₗ₀ + 2 Σ_{m=1}^{l} ℜ[K₀,ₗₘ* Yₗₘ]
	# Kθ = -√2 ℜ[ Σ_l K₁,ₗ₀*(r) Y¹ₗ₀ + Σ_{m=1}^{l} (K₁,ₗₘ* Y¹ₗₘ - K₋₁,ₗₘ* Y⁻¹ₗₘ) ] 
	# Kϕ = √2 ℑ[ Σ_l K₁,ₗ₀*(r) Y¹ₗ₀ + Σ_{m=1}^{l} (K₁,ₗₘ* Y¹ₗₘ - K₋₁,ₗₘ* Y⁻¹ₗₘ) ]

	for m in m_range(modes_lm,l)

		# Positive and negative m's are summed over, we don't need to loop over negative m's 
		# separately
		m < 0 && continue

		lm_ind = modeindex(K_all_modes,l,m)

		@views begin
		
		if iszero(m)
			@inbounds @. temp⁰ = real(conj(Kₗₘ[..,0,lm_ind])*Y[0,0])
			@inbounds @. temp⁺¹ = conj(Kₗₘ[..,1,lm_ind])*Y[0,1]
		else
			# Sum of the +m and -m terms
			@inbounds @. temp⁰ = 2real(conj(Kₗₘ[..,0,lm_ind])*Y[m,0])
			@inbounds @. temp⁺¹ = conj(Kₗₘ[..,1,lm_ind])*Y[m,1] - 
								Kₗₘ[..,-1,lm_ind]*conj(Y[m,-1])
		end

		end# views

		# r-component
		@inbounds @. K[..,1,Ω_ind] += temp⁰
		# θ-component
		@inbounds @. K[..,2,Ω_ind] += -√2*real(temp⁺¹)
		# ϕ-component
		@inbounds @. K[..,3,Ω_ind] += √2*imag(temp⁺¹)

	end
end

function Ku_longitudinal_slice_loop(ϕθ::ProductSplit,
	Kₗₘ_fits_filename,modes_lm::LM,K_all_modes::LM,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)
	
	l_min,l_max = extrema(l_range(modes_lm))
	m_min,m_max = extrema(m_range(modes_lm))

	lm_ind_min = modeindex(K_all_modes,l_min,m_min)
	lm_ind_max = modeindex(K_all_modes,l_max,m_max)

	Kₗₘ = FITS(Kₗₘ_fits_filename,"r") do f
		arr = reinterpret(ComplexF64,read(f[1]))[..,lm_ind_min:lm_ind_max]
		OffsetArray(arr,axes(arr,1),-1:1,lm_ind_min:lm_ind_max)
	end

	K = zeros(size(Kₗₘ,1),3,length(ϕθ)) # 2nd axis is for vector components (r,θ,ϕ)

	l_max = modes_lm.l_max
	dlmγ = vec(parent(zeros(-l_max:l_max, -1:1)))
	Ylmγ = vec(parent(zeros(ComplexF64, -l_max:l_max, -1:1)))

	temp⁺¹ = zeros(ComplexF64,nr)
	temp⁰ = zeros(nr)

	temparr = zeros(ComplexF64, 2l_max+1, 2l_max+1)

	for l in l_range(modes_lm)
		l == 0 && continue

		λ,v = Jy_eigen!(l,temparr)
		fill!(Ylmγ, zero(eltype(Ylmγ)))
		fill!(dlmγ, zero(eltype(dlmγ)))

		d = ClampedWignerdMatrix(l, dlmγ)
		Y = GeneralizedY(l, Ylmγ)

		for (Ω_ind,(ϕ,θ)) in enumerate(ϕθ)

			Ylmatrix!(GSH(),Y,d,l,(θ,ϕ),
				compute_d_matrix=true,λ=λ,v=v)

			populatekernel!(flows(),K,Kₗₘ,Y,modes_lm,l,
				K_all_modes,Ω_ind,temp⁰,temp⁺¹)
		
			signaltomaster!(progress_channel)
		
		end
	end
	finalize_except_wherewhence(progress_channel)

	savepath = joinpath(SCRATCH_kerneldir,"parts",
				"Klong_$(myid()).fits")
	header = FITSHeader(
		["THMIN","THMAX","PHMIN","PHMAX","PID"],
		[extrema(ϕθ,dim=2)...,extrema(ϕθ,dim=1)...,myid()],
		["Minimum colatitude",
		"Maximum colatitude",
		"Minimum longitude",
		"Maximum longitude",
		"Processor id"])
	FITS(savepath,"w") do f
		write(f,K,header=header)
	end

	return K
end

function Ku_longitudinal_slice_from_Kₗₘ(Kₗₘ_fits_filename;kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)

	header = FITS(Kₗₘ_fits_filename,"r") do f
		read_header(f[1])
	end

	K_all_modes = LM(header["L_MIN"]:header["L_MAX"],
					header["M_MIN"]:header["M_MAX"])

	l_min = get(kwargs,:l_min,K_all_modes.l_min); header["L_MIN"] = l_min;
	l_max = get(kwargs,:l_max,K_all_modes.l_max); header["L_MAX"] = l_max;
	m_min = get(kwargs,:m_min,min(K_all_modes.m_min,l_max)); header["M_MIN"] = m_min;
	m_max = get(kwargs,:m_max,min(K_all_modes.m_max,l_max)); header["M_MAX"] = m_max;
	kernel_modes = LM(l_min:l_max,m_min:m_max)
	nl = length(l_min:l_max)

	nθ_default = max(360,4header["L_MAX"])
	nθ = get(kwargs,:nθ,nθ_default)
	θ_default = LinRange(0,π,nθ)
	θ = get(kwargs,:θ,θ_default)
	nθ = length(θ)
	ϕ = get(kwargs,:ϕ,(header["PHI1"] + header["PHI2"])/2)
	nl = length(l_range(kernel_modes))

	ϕθ = Iterators.product(ϕ:ϕ,θ)

	Krθ = pmapreduce_timed(Ku_longitudinal_slice_loop,x->cat(x...,dims=3),
		ϕθ,Kₗₘ_fits_filename,kernel_modes,K_all_modes;
		progress_str="Modes summed in kernel: ",nprogressticks=nθ*nl)

	Krθ = permutedims(Krθ,[1,3,2]) # last axis will store the vector components

	# Write to fits file

	header["THSLMN"] = minimum(θ)
	set_comment!(header,"THSLMN","Minimum colatitude in slice")
	
	header["THSLMX"] = maximum(θ)
	set_comment!(header,"THSLMX","Maximum colatitude in slice")

	header["NTH"] = nθ
	set_comment!(header,"NTH","Number of grid points in colatitude")

	header["PHSL"] = ϕ
	set_comment!(header,"PHSL","Azimuth at which the slice is computed")

	FITS(joinpath(SCRATCH_kerneldir,"Ku_rθslice_from_Klm.fits"),"w") do f
		write(f,Krθ,header=header)
	end
	Krθ
end

function Ku_latitudinal_slice_loop(ϕθ::ProductSplit,
	Kₗₘ_fits_filename,modes_lm::LM,K_all_modes::LM,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	l_min,l_max = extrema(l_range(modes_lm))
	m_min,m_max = extrema(m_range(modes_lm))

	lm_ind_min = modeindex(K_all_modes,l_min,m_min)
	lm_ind_max = modeindex(K_all_modes,l_max,m_max)

	Kₗₘ = FITS(Kₗₘ_fits_filename,"r") do f
		arr = reinterpret(ComplexF64,read(f[1]))[..,lm_ind_min:lm_ind_max]
		OffsetArray(arr,axes(arr,1),-1:1,lm_ind_min:lm_ind_max)
	end

	K = zeros(size(Kₗₘ,1),3,length(ϕθ)) # 2nd axis is for vector components (r,θ,ϕ)
	
	l_max = modes_lm.l_max
	dlmγ = vec(parent(zeros(-l_max:l_max,-1:1)))
	Ylmγ = vec(parent(zeros(ComplexF64,-l_max:l_max,-1:1)))

	temp⁺¹ = zeros(ComplexF64,nr)
	temp⁰ = zeros(nr)

	temparr = zeros(ComplexF64, 2l_max+1, 2l_max+1)

	θ = first(ϕθ)[2] # θ is fixed

	for l in l_range(modes_lm)
		l == 0 && continue

		fill!(Ylmγ, zero(eltype(Ylmγ)))
		fill!(dlmγ, zero(eltype(dlmγ)))
		d = ClampedWignerdMatrix(l,dlmγ)
		Y = GeneralizedY(l,Ylmγ)
		djmatrix!(d,l,θ,temparr)
		
		for (Ω_ind,(ϕ,θ)) in enumerate(ϕθ)
			
			Ylmatrix!(GSH(),Y,d,l,(θ,ϕ),compute_d_matrix=false)

			populatekernel!(flows(),K,Kₗₘ,Y,modes_lm,l,
				K_all_modes,Ω_ind,temp⁰,temp⁺¹)

			signaltomaster!(progress_channel)

		end
	end
	finalize_except_wherewhence(progress_channel)

	savepath = joinpath(SCRATCH_kerneldir,"parts",
				"Klat_$(myid()).fits")

	header = FITSHeader(
		["THMIN","THMAX","PHMIN","PHMAX","PID"],
		[extrema(ϕθ,dim=2)...,extrema(ϕθ,dim=1)...,myid()],
		["Minimum colatitude",
		"Maximum colatitude",
		"Minimum longitude",
		"Maximum longitude",
		"Processor id"])
	FITS(savepath,"w") do f
		write(f,K,header=header)
	end

	return K
end

function Ku_latitudinal_slice_from_Kₗₘ(Kₗₘ_fits_filename;kwargs...)
	
	r_src = get(kwargs,:r_src,r_src_default)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)

	header = FITS(Kₗₘ_fits_filename,"r") do f
		read_header(f[1])
	end

	K_all_modes = LM(header["L_MIN"]:header["L_MAX"],
					header["M_MIN"]:header["M_MAX"])
	
	l_min = get(kwargs,:l_min,K_all_modes.l_min); header["L_MIN"] = l_min;
	l_max = get(kwargs,:l_max,K_all_modes.l_max); header["L_MAX"] = l_max;
	m_min = get(kwargs,:m_min,min(K_all_modes.m_min,l_max)); header["M_MIN"] = m_min;
	m_max = get(kwargs,:m_max,min(K_all_modes.m_max,l_max)); header["M_MAX"] = m_max;
	kernel_modes = LM(l_min:l_max,m_min:m_max)
	nl = length(l_min:l_max)

	θ = get(kwargs,:θ,(header["TH1"] + header["TH2"])/2)
	nϕ_default = max(720,4header["L_MAX"])
	nϕ = get(kwargs,:nϕ,nϕ_default)
	ϕ_default = LinRange(0,2π,nϕ)
	ϕ = get(kwargs,:ϕ,ϕ_default)
	nϕ = length(ϕ)
	nl = length(l_range(kernel_modes))

	ϕθ = Iterators.product(ϕ,θ:θ)

	Krϕ = pmapreduce_timed(Ku_latitudinal_slice_loop,x->cat(x...,dims=3),
		ϕθ,Kₗₘ_fits_filename,kernel_modes,K_all_modes;
		progress_str="Modes summed in kernel: ",nprogressticks=nϕ*nl)

	Krϕ = permutedims(Krϕ,[1,3,2]) # last axis will store the vector components

	header["PHSLMN"] = minimum(ϕ)
	set_comment!(header,"PHSLMN","Minimum azimuth in slice")
	
	header["PHSLMX"] = maximum(ϕ)
	set_comment!(header,"PHSLMX","Maximum azimuth in slice")

	header["NPHI"] = nϕ
	set_comment!(header,"NPHI","Number of grid points in azimuth")

	header["THSL"] = θ
	set_comment!(header,"THSL","Colatitude at which the slice is computed")
	
	FITS(joinpath(SCRATCH_kerneldir,"Ku_rϕslice_from_Klm.fits"),"w") do f
		write(f,Krϕ,header=header)
	end
	Krϕ
end

function Ku_surface_slice_loop(ϕθ::ProductSplit,
	Kₗₘ_fits_filename,modes_lm::LM,K_all_modes::LM,
	r_obs_ind,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	l_min,l_max = extrema(l_range(modes_lm))
	m_min,m_max = extrema(m_range(modes_lm))

	lm_ind_min = modeindex(K_all_modes,l_min,m_min)
	lm_ind_max = modeindex(K_all_modes,l_max,m_max)

	Kₗₘ = FITS(Kₗₘ_fits_filename,"r") do f
		arr = reinterpret(ComplexF64,read(f[1]))[r_obs_ind,:,lm_ind_min:lm_ind_max]
		OffsetArray(arr,-1:1,lm_ind_min:lm_ind_max)
	end

	K = zeros(3,length(ϕθ))

	l_max = modes_lm.l_max
	dlmγ = vec(parent(zeros(-l_max:l_max,-1:1)))
	Ylmγ = vec(parent(zeros(ComplexF64,-l_max:l_max,-1:1)))

	θ_prev = first(ϕθ)[2]

	temp⁺¹ = zeros(ComplexF64)
	temp⁰ = zeros()

	temparr = zeros(ComplexF64,2l_max+1,2l_max+1)

	for l in l_range(modes_lm)
		l == 0 && continue

		fill!(Ylmγ, zero(eltype(Ylmγ)))
		fill!(dlmγ, zero(eltype(dlmγ)))
		d = ClampedWignerdMatrix(l,dlmγ)
		Y = GeneralizedY(l,Ylmγ)
	
		for (Ω_ind,(ϕ,θ)) in enumerate(ϕθ)

			if (θ != θ_prev) || (Ω_ind == 1)
				djmatrix!(d,l,θ,temparr)
				θ_prev = θ
			end

			Ylmatrix!(GSH(),Y,d,l,(θ,ϕ),compute_d_matrix=false)

			populatekernel!(flows(),K,Kₗₘ,Y,modes_lm,l,
				K_all_modes,Ω_ind,temp⁰,temp⁺¹)
			
			signaltomaster!(progress_channel)

		end
	end
	finalize_except_wherewhence(progress_channel)

	savepath = joinpath(SCRATCH_kerneldir,"parts",
				"Ksurf_$(myid()).fits")
	header = FITSHeader(
		["THMIN","THMAX","PHMIN","PHMAX","PID"],
		[extrema(ϕθ,dim=2)...,extrema(ϕθ,dim=1)...,myid()],
		["Minimum colatitude",
		"Maximum colatitude",
		"Minimum longitude",
		"Maximum longitude",
		"Processor id"])
	FITS(savepath,"w") do f
		write(f,K,header=header)
	end

	return K
end

function Ku_surface_slice_from_Kₗₘ(Kₗₘ_fits_filename;kwargs...)
	
	r_src = get(kwargs,:r_src,r_src_default)
	r_obs = get(kwargs,:r_obs,r_obs_default)

	r_obs_ind = radial_grid_index(r_obs)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)

	header = FITS(Kₗₘ_fits_filename,"r") do f
		read_header(f[1])
	end

	K_all_modes = LM(header["L_MIN"]:header["L_MAX"],
					header["M_MIN"]:header["M_MAX"])
	
	l_min = get(kwargs,:l_min,K_all_modes.l_min); header["L_MIN"] = l_min;
	l_max = get(kwargs,:l_max,K_all_modes.l_max); header["L_MAX"] = l_max;
	m_min = get(kwargs,:m_min,min(K_all_modes.m_min,l_max)); header["M_MIN"] = m_min;
	m_max = get(kwargs,:m_max,min(K_all_modes.m_max,l_max)); header["M_MAX"] = m_max;
	kernel_modes = LM(l_min:l_max,m_min:m_max)
	nl = length(l_min:l_max)

	nϕ_default = max(720,4header["L_MAX"])
	nϕ = get(kwargs,:nϕ,nϕ_default)
	ϕ_default = LinRange(0,2π,nϕ)
	ϕ = get(kwargs,:ϕ,ϕ_default)
	nϕ = length(ϕ)

	nθ_default = max(360,4header["L_MAX"])
	nθ = get(kwargs,:nθ,nθ_default)
	θ_default = LinRange(0,π,nθ)
	θ = get(kwargs,:θ,θ_default)
	nθ = length(θ)
	nl = length(l_range(kernel_modes))

	ϕθ = Iterators.product(ϕ,θ)

	K = pmapreduce_timed(Ku_surface_slice_loop,x->hcat(x...),ϕθ,
			Kₗₘ_fits_filename,kernel_modes,K_all_modes,r_obs_ind;
			progress_str="Modes summed in kernel: ",nprogressticks=nϕ*nθ*nl)

	Kϕθ = permutedims(reshape(K,3,nϕ,nθ),[2,3,1]) # last axis will store the vector components

	header["PHSLMN"] = minimum(ϕ)
	set_comment!(header,"PHSLMN","Minimum azimuth in slice")
	
	header["PHSLMX"] = maximum(ϕ)
	set_comment!(header,"PHSLMX","Maximum azimuth in slice")

	header["NPHI"] = nϕ
	set_comment!(header,"NPHI","Number of grid points in azimuth")

	header["THSLMN"] = minimum(θ)
	set_comment!(header,"THSLMN","Minimum colatitude in slice")
	
	header["THSLMX"] = maximum(θ)
	set_comment!(header,"THSLMX","Maximum colatitude in slice")

	header["NTH"] = nθ
	set_comment!(header,"NTH","Number of grid points in colatitude")

	header["ROBS"] = r_obs
	set_comment!(header,"ROBS","Observation radius")
	
	FITS(joinpath(SCRATCH_kerneldir,"Ku_ϕθ_from_Klm.fits"),"w") do f
		write(f,Kϕθ,header=header)
	end
	Kϕθ
end

#######################################################################

function populatekernel!(::soundspeed,K,Kₗₘ,Ylm,modes_lm,Ω_ind)

	# The kernel is defined as K = Σ_l Kₗ₀(r) Yₗ₀ + 2 Σ_{m=1}^{l} ℜ[Kₗₘ Yₗₘ]

	for (lm_ind,(l,m)) in enumerate(modes_lm)

		# don't need to iterate over negative m's as they are already summed over
		(m < 0) && continue

		# lm_ind = modeindex(modes_lm,(l,m))
		SH_lm_ind = SphericalHarmonics.index_y(l,m)

		@views begin
		if iszero(m)
			# Term is purely real
			@inbounds @. K[..,Ω_ind] += real(Kₗₘ[..,lm_ind])*real(Ylm[SH_lm_ind])
		else
			# Sum of the +t and -t terms
			@inbounds @. K[..,Ω_ind] += 2real(Kₗₘ[..,lm_ind]*Ylm[SH_lm_ind])
		end
		end #views
	end
end

function Kc_angular_slice_loop(ϕθ::ProductSplit,
	Kst_fits_filename,st_iterator::LM,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	s_max = st_iterator.l_max

	P = SphericalHarmonics.allocate_p(s_max)
	Yst = SphericalHarmonics.allocate_y(s_max)

	coeff = SphericalHarmonics.compute_coefficients(s_max)

	Kst = FITS(Kst_fits_filename,"r") do f
		reinterpret(ComplexF64,read(f[1]))
	end

	K = zeros(size(Kst,1),length(ϕθ))

	θ_prev = first(ϕθ)[2]

	for (Ω_ind,(ϕ,θ)) in enumerate(ϕθ)

		if (θ != θ_prev) || (Ω_ind == 1)
			compute_p!(s_max,cos(θ),coeff,P)
			θ_prev == θ
		end
		compute_y!(s_max,cos(θ),ϕ,P,Yst)

		populatekernel!(soundspeed(),K,Kst,Yst,st_iterator,Ω_ind)

		signaltomaster!(progress_channel)
	end
	finalize_except_wherewhence(progress_channel)
	return K
end

function Kc_longitudinal_slice_from_Kₗₘ(Kst_fits_filename;kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)

	@load(joinpath(Gfn_path_src,"parameters.jld2"),ℓ_arr)

	header = FITS(Kst_fits_filename,"r") do f
		read_header(f[1])
	end

	s_max = header["L_MAX"]
	t_max = header["M_MAX"]
	s_min = header["L_MIN"]
	t_min = header["M_MIN"]
	st_iterator = LM(s_min:s_max,t_min:t_max)

	θ = get(kwargs,:θ,LinRange(0,π,get(kwargs,:nθ,4s_max)))
	nθ = length(θ)

	header["THSLMN"] = minimum(θ)
	set_comment!(header,"THSLMN","Minimum colatitude in slice")
	
	header["THSLMX"] = maximum(θ)
	set_comment!(header,"THSLMX","Maximum colatitude in slice")

	header["NTH"] = nθ
	set_comment!(header,"NTH","Number of grid points in colatitude")

	ϕ = get(kwargs,:ϕ,(header["PHI1"] + header["PHI2"])/2)
	
	header["PHSL"] = ϕ
	set_comment!(header,"PHSL","Azimuth at which the slice is computed")

	ϕθ = Iterators.product(ϕ:ϕ,θ)

	Krθ = pmapreduce_timed(Kc_angular_slice_loop,x->hcat(x...),ϕθ,
			Kst_fits_filename,st_iterator;
			progress_str = "Angles processed : ")
	
	FITS(joinpath(SCRATCH_kerneldir,"Krθ_from_Kst.fits"),"w") do f
		write(f,Krθ,header=header)
	end
	Krθ
end

function Kc_latitudinal_slice_from_Kₗₘ(Kst_fits_filename;kwargs...)
	
	r_src = get(kwargs,:r_src,r_src_default)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)

	@load(joinpath(Gfn_path_src,"parameters.jld2"),ℓ_arr)

	header = FITS(Kst_fits_filename,"r") do f
		read_header(f[1])
	end

	s_max = header["L_MAX"]
	t_max = header["M_MAX"]
	s_min = header["L_MIN"]
	t_min = header["M_MIN"]
	st_iterator = LM(s_min:s_max,t_min:t_max)

	ϕ = get(kwargs,:ϕ,LinRange(0,2π,get(kwargs,:nϕ,4s_max)))
	nϕ = length(ϕ)

	header["PHSLMN"] = minimum(ϕ)
	set_comment!(header,"PHSLMN","Minimum azimuth in slice")
	
	header["PHSLMX"] = maximum(ϕ)
	set_comment!(header,"PHSLMX","Maximum azimuth in slice")

	header["NPHI"] = nϕ
	set_comment!(header,"NPHI","Number of grid points in azimuth")

	θ = get(kwargs,:θ,(header["TH1"] + header["TH2"])/2)
	header["THSL"] = θ
	set_comment!(header,"THSL","Colatitude at which the slice is computed")

	ϕθ = Iterators.product(ϕ,θ:θ)

	Krϕ = pmapreduce_timed(Kc_angular_slice_loop,x->hcat(x...),ϕθ,
			Kst_fits_filename,st_iterator;
			progress_str = "Angles processed : ")
	
	FITS(joinpath(SCRATCH_kerneldir,"Krϕ_from_Kst.fits"),"w") do f
		write(f,Krϕ,header=header)
	end
	Krϕ
end

function Kc_surface_slice_loop(ϕθ::ProductSplit,
	Kst_fits_filename,st_iterator,r_obs_ind,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	s_max = st_iterator.l_max

	P = SphericalHarmonics.allocate_p(s_max)
	Yst = SphericalHarmonics.allocate_y(s_max)

	coeff = SphericalHarmonics.compute_coefficients(s_max)

	Kst = FITS(Kst_fits_filename,"r") do f
		reinterpret(ComplexF64,read(f[1]))[r_obs_ind,:]
	end

	K = zeros(length(ϕθ))

	θ_prev = first(ϕθ)[2]
	
	for (Ω_ind,(ϕ,θ)) in enumerate(ϕθ)

		if (θ != θ_prev) || (Ω_ind == 1)
			compute_p!(s_max,cos(θ),coeff,P)
			θ_prev = θ
		end

		compute_y!(s_max,cos(θ),ϕ,P,Yst)

		populatekernel!(soundspeed(),K,Kst,Yst,st_iterator,Ω_ind)

		signaltomaster!(progress_channel)
	end
	finalize_except_wherewhence(progress_channel)
	return K
end

function Kc_surface_slice_from_Kₗₘ(Kst_fits_filename;kwargs...)
	
	r_src = get(kwargs,:r_src,r_src_default)
	r_obs = get(kwargs,:r_obs,r_obs_default)

	r_obs_ind = radial_grid_index(r_obs)

	Gfn_path_src,Gfn_path_obs1,Gfn_path_obs2 = 
		Gfn_path_from_source_radius.((r_src,xobs1.r,xobs2.r))

	@load(joinpath(Gfn_path_src,"parameters.jld2"),ℓ_arr)

	header = FITS(Kst_fits_filename,"r") do f
		read_header(f[1])
	end

	s_max = header["L_MAX"]
	t_max = header["M_MAX"]
	s_min = header["L_MIN"]
	t_min = header["M_MIN"]
	st_iterator = LM(s_min:s_max,t_min:t_max)

	ϕ = get(kwargs,:ϕ,LinRange(0,2π,get(kwargs,:nϕ,2s_max)))
	nϕ = length(ϕ)
	θ = get(kwargs,:θ,LinRange(0,π,get(kwargs,:nθ,2s_max)))
	nθ = length(θ)

	ϕθ = Iterators.product(ϕ,θ)

	K = pmapreduce_timed(Kc_surface_slice_loop,x->hcat(x...),ϕθ,
			Kst_fits_filename,st_iterator,r_obs_ind;
			progress_str = "Angles processed : ")

	Kϕθ = collect(reshape(K,nϕ,nθ))

	header["PHSLMN"] = minimum(ϕ)
	set_comment!(header,"PHSLMN","Minimum azimuth in slice")
	
	header["PHSLMX"] = maximum(ϕ)
	set_comment!(header,"PHSLMX","Maximum azimuth in slice")

	header["NPHI"] = nϕ
	set_comment!(header,"NPHI","Number of grid points in azimuth")

	header["THSLMN"] = minimum(θ)
	set_comment!(header,"THSLMN","Minimum colatitude in slice")
	
	header["THSLMX"] = maximum(θ)
	set_comment!(header,"THSLMX","Maximum colatitude in slice")

	header["NTH"] = nθ
	set_comment!(header,"NTH","Number of grid points in colatitude")

	header["ROBS"] = r_obs
	set_comment!(header,"ROBS","Observation radius")
	
	FITS(joinpath(SCRATCH_kerneldir,"Kϕθ_from_Kst.fits"),"w") do f
		write(f,Kϕθ,header=header)
	end
	Kϕθ
end

#######################################################################
# The following function compute the 3D kernel using the fleshed-out formulation
# and not from the spherical harmonic coefficients
# They are only defined for radial displacements, and are used to compare 
# our results with those obtained using a formalism similar to Mandal et al. (2017)
#######################################################################

@inline function populatekernel3D!(::soundspeed,K,Ω_ind,
	realconjhωHjₒjₛω_r₁r₂,Pjₒ1_Pjₛ2,realconjhωconjHjₒjₛω_r₂r₁,Pjₒ2_Pjₛ1,pre)

	@inbounds P12 = Pjₒ1_Pjₛ2[Ω_ind]
	@inbounds P21 = Pjₒ2_Pjₛ1[Ω_ind]

	iszero(P12) && iszero(P21) && return
	
	@inbounds for r_ind in axes(K,1)
		K[r_ind,Ω_ind] += pre * (
						realconjhωHjₒjₛω_r₁r₂[r_ind] * P12  + 
						realconjhωconjHjₒjₛω_r₂r₁[r_ind] * P21 )
	end
end

function Kc_partial(modes_iter_proc::ProductSplit,
	xobs1,xobs2,los::los_radial,θ_arr,ϕ_arr,hω_arr,
	p_Gsrc::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing,ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing,ParamsGfn}=p_Gobs1,
	r_src=r_src_default,r_obs=nothing,
	progress_channel::Union{Nothing,RemoteChannel}=nothing,
	timers_channel::Union{Nothing,RemoteChannel}=nothing)

	localtimer = TimerOutput()

	p_Gsrc = read_all_parameters(p_Gsrc,r_src=r_src)
	p_Gobs1 = read_all_parameters(p_Gobs1,r_src=xobs1.r)
	p_Gobs2 = read_all_parameters(p_Gobs2,r_src=xobs2.r)

	Gfn_path_src,NGfn_files_src = p_Gsrc.path,p_Gsrc.num_procs
	Gfn_path_obs1,NGfn_files_obs1 =  p_Gobs1.path,p_Gobs1.num_procs
	Gfn_path_obs2,NGfn_files_obs2 = p_Gobs2.path,p_Gobs2.num_procs
	@unpack ℓ_arr,ω_arr,Nν_Gfn,dω = p_Gsrc

	r₁_ind,r₂_ind = radial_grid_index.((xobs1,xobs2))
	obs_at_same_height = r₁_ind == r₂_ind

	nθ,nϕ = length(θ_arr),length(ϕ_arr)
	nΩ = nθ * nϕ
	θϕ_iter = Iterators.product(θ_arr,ϕ_arr)

	# jₛ is the angular degree of the wave from the source, and 
	# jₒ is the angular degree of the wave from the observation point
	jₛ_min_proc,jₛ_max_proc = extrema(modes_iter_proc,1)
	νind_min_proc,νind_max_proc = extrema(modes_iter_proc,2)
	j_min,j_max = extrema(ℓ_arr)
	jₛ_range_proc = jₛ_min_proc:jₛ_max_proc
	jₒjₛ_allmodes = L₂L₁Δ(jₛ_range_proc,2j_max,j_min:j_max)
	jₒ_range_proc = l₂_range(jₒjₛ_allmodes)

	# Need to load all the modes on this processor
	modes_obsGfn_proc = ProductSplit((jₒ_range_proc,νind_min_proc:νind_max_proc),1,1)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr,1:Nν_Gfn),modes_iter_proc,NGfn_files_src)

	# Gℓ′ω(r,robs) files
	Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_obs1,
						(ℓ_arr,1:Nν_Gfn),modes_obsGfn_proc,NGfn_files_obs1)

	Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_obs2,
						(ℓ_arr,1:Nν_Gfn),modes_obsGfn_proc,NGfn_files_obs2)

	K = zeros(nr,nΩ)

	arrs = kernel.allocatearrays(soundspeed(),los,obs_at_same_height)
	@unpack Gsrc,drGsrc,Gobs1,drGobs1,Gobs2,drGobs2,divGsrc,divGobs = arrs
	@unpack fjₒjₛ_r₁_rsrc,fjₒjₛ_r₂_rsrc,Hjₒjₛω_r₁r₂,Hjₒjₛω_r₂r₁ = arrs
	@unpack Gobs1_cache,drGobs1_cache,Gobs2_cache,drGobs2_cache = arrs
	@unpack tworealconjhωHjₒjₛω_r₁r₂,tworealconjhωconjHjₒjₛω_r₂r₁ = arrs

	@unpack Gobs1_cache,Gobs2_cache = arrs
	for jₒ in jₒ_range_proc
		Gobs1_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(Gsrc)...)
		drGobs1_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(drGsrc)...)
		if !obs_at_same_height
			Gobs2_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(Gsrc)...)
			drGobs2_cache[jₒ] = zeros_Float64_to_ComplexF64(axes(drGsrc)...)
		end
	end

	@timeit localtimer "legendre" begin
	P1,P2 = zeros(0:j_max,nΩ),zeros(0:j_max,nΩ)
	for (Ω_ind,(θ,ϕ)) in enumerate(θϕ_iter)
		Pl!(view(P1,:,Ω_ind),cosχ((θ,ϕ),xobs1),lmax=j_max)
		Pl!(view(P2,:,Ω_ind),cosχ((θ,ϕ),xobs2),lmax=j_max)
	end

	P1 = permutedims(P1)
	P2 = permutedims(P2)
	end # timer

	Pjₒ1_Pjₛ2 = zeros(nΩ)
	Pjₒ2_Pjₛ1 = zeros(nΩ)

	ω_ind_prev = -1
	for (jₛ,ω_ind) in modes_iter_proc

		ω = ω_arr[ω_ind]
		conjhω = conj(hω_arr[ω_ind])
		dωω²Pω = dω/2π * ω^2 * Powspec(ω)

		@timeit localtimer "FITS" begin
		    
		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(jₛ,ω_ind),NGfn_files_src,:,1:2,1,1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
			(ℓ_arr,1:Nν_Gfn),(jₛ,ω_ind),NGfn_files_src,:,1:1,1,2)

		end # timer

		Grr_r₁_rsrc = Gsrc[r₁_ind,0]
		Grr_r₂_rsrc = Gsrc[r₂_ind,0]

		jₒ_range_jₛ = l₂_range(jₒjₛ_allmodes,jₛ)

		for jₒ in jₒ_range_jₛ

			pre = dωω²Pω * (2jₒ+1)/4π * (2jₛ+1)/4π

			Gobs1 = Gobs1_cache[jₒ]
			drGobs1 = drGobs1_cache[jₒ]
			Gobs2 = Gobs2_cache[jₒ]
			drGobs2 = drGobs2_cache[jₒ]

			@timeit localtimer "FITS" begin
			    
			# Green function about receiver location
			if ω_ind != ω_ind_prev || jₒ ∉ l₂_range(jₒjₛ_allmodes,jₛ-1)
				read_Gfn_file_at_index!(Gobs1,Gfn_fits_files_obs1,
					(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs1,:,1:2,1,1)

				# Derivative of Green function about receiver location
				read_Gfn_file_at_index!(drGobs1,Gfn_fits_files_obs1,
					(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs1,:,1:1,1,2)
			end
			end # timer

			@timeit localtimer "radial term" begin
			# precompute the radial term in f
			radial_fn_δc_firstborn!(fjₒjₛ_r₁_rsrc,Gsrc,drGsrc,jₛ,divGsrc,
				Gobs1,drGobs1,jₒ,divGobs)
			end # timer

			@timeit localtimer "radial term 2" begin
			Hjₒjₛω!(Hjₒjₛω_r₁r₂,fjₒjₛ_r₁_rsrc,Grr_r₂_rsrc)
			end
			@timeit localtimer "radial term 3" begin
			@. tworealconjhωHjₒjₛω_r₁r₂ = 2real(conjhω * Hjₒjₛω_r₁r₂)
			end

			if r₁_ind != r₂_ind
				@timeit localtimer "FITS" begin
				if ω_ind != ω_ind_prev || jₒ ∉ l₂_range(jₒjₛ_allmodes,jₛ-1)
					# Green function about receiver location
					read_Gfn_file_at_index!(Gobs2,Gfn_fits_files_obs2,
						(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs2,:,1:2,1,1)

					# Derivative of Green function about receiver location
					read_Gfn_file_at_index!(drGobs2,Gfn_fits_files_obs2,
						(ℓ_arr,1:Nν_Gfn),(jₒ,ω_ind),NGfn_files_obs2,:,1:1,1,2)
				end
				end # timer

				@timeit localtimer "radial term" begin
				radial_fn_δc_firstborn!(fjₒjₛ_r₂_rsrc,Gsrc,drGsrc,jₛ,divGsrc,
					Gobs2,drGobs2,jₒ,divGobs)
				
				end # timer
			end
			@timeit localtimer "radial term 2" begin
			Hjₒjₛω!(Hjₒjₛω_r₂r₁,fjₒjₛ_r₂_rsrc,Grr_r₁_rsrc)
			end
			@timeit localtimer "radial term 3" begin
			@. tworealconjhωconjHjₒjₛω_r₂r₁ = 2real(conjhω * conj(Hjₒjₛω_r₂r₁))
			end

			@timeit localtimer "legendre" begin
			@inbounds for Ωind in eachindex(Pjₒ1_Pjₛ2)
				Pjₒ1_Pjₛ2[Ωind] = P1[Ωind,jₒ]*P2[Ωind,jₛ]
				Pjₒ2_Pjₛ1[Ωind] = P1[Ωind,jₛ]*P2[Ωind,jₒ]
			end
			end # timer

			@timeit localtimer "kernel" begin
			for Ω_ind in axes(K,2)
				populatekernel3D!(soundspeed(),K,Ω_ind,
				tworealconjhωHjₒjₛω_r₁r₂,Pjₒ1_Pjₛ2,
				tworealconjhωconjHjₒjₛω_r₂r₁,Pjₒ2_Pjₛ1,
				pre)
			end
			end # timer
		end

		ω_ind_prev = ω_ind

		signaltomaster!(progress_channel)
	end

	map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs1,Gfn_fits_files_obs2))

	signaltomaster!(timers_channel,localtimer)
	map(finalize_except_wherewhence,(progress_channel,timers_channel))

	return K
end

@def generatefitsheader begin
    header = FITSHeader(["r1","th1","phi1","r2","th2","phi2",
		"l_max","m_max","jmin","jmax","nuimin","nuimax",
		"PHSLMN","PHSLMX","NPHI",
		"THSLMN","THSLMX","NTH"],
		Any[float(xobs1.r),float(xobs1.θ),float(xobs1.ϕ),
		float(xobs2.r),float(xobs2.θ),float(xobs2.ϕ),Int(s_max),Int(t_max),
		minimum(ℓ_range),maximum(ℓ_range),minimum(ν_ind_range),maximum(ν_ind_range),
		minimum(ϕ),maximum(ϕ),length(ϕ),
		minimum(θ),maximum(θ),length(θ)],
		["Radius of the first observation point",
		"Colatitude of the first observation point",
		"Azimuth of the first observation point",
		"Radius of the second observation point",
		"Colatitude of the second observation point",
		"Azimuth of the second observation point",
		"Maximum angular degree of the perturbation",
		"Maximum azimuthal order of the perturbation",
		"Minimum wave mode","Maximum wave mode",
		"Minimum wave frequency index","Maximum wave frequency index",
		"Minimum azimuth in slice","Maximum azimuth in slice",
		"Number of grid points in azimuth",
		"Minimum colatitude in slice","Maximum colatitude in slice",
		"Number of grid points in colatitude"])
end

function _Kc(xobs1::Point3D,xobs2::Point3D,los::los_direction;θ,ϕ,kwargs...)
	r_src,r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc,p_Gobs1,p_Gobs2 = read_parameters_for_points(xobs1,xobs2;kwargs...)

	@unpack ν_arr,ℓ_arr,num_procs,dω,ν_start_zeros,Nν_Gfn = p_Gsrc
	ℓ_range,ν_ind_range,modes_iter,np = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)

	hω_arr = get(kwargs,:hω) do
		hω(TravelTimes(),xobs1,xobs2,los;kwargs...,
			ℓ_range=ℓ_range,ν_ind_range=ν_ind_range,
			print_timings=false)
	end

	pmapsum_timed(Kc_partial,modes_iter,
				xobs1,xobs2,los,θ,ϕ,hω_arr,
				p_Gsrc,p_Gobs1,p_Gobs2,r_src,r_obs;				
				progress_str="Modes summed in 3D kernel : ",kwargs...)
end

function Kc_latitudinal_slice(xobs1::Point3D,xobs2::Point3D,los::los_radial=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack ℓ_arr,ν_arr = p_Gsrc
	ℓ_range,ν_ind_range = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)
	ℓmin,ℓmax = extrema(ℓ_range)
	# These are derived form ℓmax
	s_max = 2ℓmax; t_max = s_max

	ϕ = get(kwargs,:ϕ,LinRange(0,2π,get(kwargs,:nϕ,4ℓmax)))
	θ = get(kwargs,:θ,(xobs1.θ+xobs2.θ)/2)

	Kc_3D_lat_slice = _Kc(xobs1,xobs2,los;kwargs...,θ=θ,ϕ=ϕ)
	
	@generatefitsheader

	filename = joinpath(SCRATCH_kerneldir,
		"Kc_3D_lat_slice_jcutoff$(ℓmax).fits")

	FITS(filename,"w") do f
		write(f,Kc_3D_lat_slice,header=header)
	end
	Kc_3D_lat_slice
end

function Kc_longitudinal_slice(xobs1::Point3D,xobs2::Point3D,los::los_radial=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack ℓ_arr,ν_arr = p_Gsrc
	ℓ_range,ν_ind_range = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)
	ℓmin,ℓmax = extrema(ℓ_range)
	# These are derived form ℓmax
	s_max = 2ℓmax; t_max = s_max

	θ = get(kwargs,:θ,LinRange(0,π,get(kwargs,:nθ,4ℓmax)))
	ϕ = get(kwargs,:ϕ,(xobs1.ϕ+xobs2.ϕ)/2)

	Kc_3D_long_slice = _Kc(xobs1,xobs2,los;kwargs...,θ=θ,ϕ=ϕ)
	
	@generatefitsheader

	filename = joinpath(SCRATCH_kerneldir,
		"Kc_3D_long_slice_jcutoff$(ℓmax).fits")

	FITS(filename,"w") do f
		write(f,Kc_3D_long_slice,header=header)
	end
	Kc_3D_long_slice
end

# 3D profile
function Kc(xobs1::Point3D,xobs2::Point3D,los::los_radial=los_radial();kwargs...)
	
	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack ℓ_arr,ν_arr = p_Gsrc
	ℓ_range,ν_ind_range = ℓ_and_ν_range(ℓ_arr,ν_arr;kwargs...)
	ℓmin,ℓmax = extrema(ℓ_range)
	# These may be derived form ℓmax

	s_max = get(kwargs,:s_max,2ℓmax); 
	t_max = get(kwargs,:t_max,s_max);

	# A polynomial of degree 2N-1 can be integrated using N Gauss-Legendre 
	# nodes. We treat SH of angular degree s as a polynomial of the same degree
	Nnodes = get(kwargs,:nθ,ceil(Int,(s_max+1)/2))
	nodes,_ = gausslegendre(Nnodes)
	θdefault = acos.(nodes)[end:-1:1] # flipped to get to increasing θ from increasing cosθ

	θ = get(kwargs,:θ,θdefault)
	ϕ = get(kwargs,:ϕ,LinRange(0,2π,get(kwargs,:nϕ,2s_max+1)))

	Kc_3D = _Kc(xobs1,xobs2,los;kwargs...,θ=θ,ϕ=ϕ)
	Kc_3D = copy(reshape(Kc_3D,nr,length(θ),length(ϕ)))
	
	@generatefitsheader

	filename = joinpath(SCRATCH_kerneldir,
		"Kc_3D_jcutoff$(ℓmax).fits")

	FITS(filename,"w") do f
		write(f,Kc_3D,header=header)
	end
	Kc_3D
end

#######################################################################
# The following functions use the displacement and not the cross-covariance
# This is to compare with the results obtained by Mandal et al. (2017)
#######################################################################

function Kc_latitudinal_slice_Mandal(xobs::Point3D,xsrc::Point3D;kwargs...)
	
	# Slice at a constant θ
	θ = get(kwargs,:θ,(xobs.θ + xsrc.θ)/2)

	Gfn_path_src,Gfn_path_obs = 
		Gfn_path_from_source_radius.((xsrc.r,xobs.r))

	@load(joinpath(Gfn_path_src,"parameters.jld2"),
		ω_arr,ℓ_arr,num_procs,dω,ν_start_zeros,Nν_Gfn)

	num_procs_obs = get_numprocs(Gfn_path_obs)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr); lmax = maximum(ℓ_range)
	ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
	modes_iter = Iterators.product(ℓ_range,ℓ_range,ν_ind_range)
	ℓω_iter = Iterators.product(ℓ_range,ν_ind_range)

	if haskey(kwargs,:hω)
		hω = kwargs.hω
	else
		hω = hω(xobs,xsrc;kwargs...)
	end
	hω = OffsetVector(hω[ν_start_zeros .+ ν_ind_range],ν_ind_range)

	nϕ = get(kwargs,:nϕ,2ℓmax)
	ϕ_arr = get(kwargs,:ϕ,LinRange(0,2π,nϕ))
	nϕ = length(ϕ_arr)

	function summodes(modes_iter_proc)

		m = moderanges_common_lastarray(modes_iter_proc)
		
		j₁ω_min,j₂ω_min = first(m)
		j₁ω_max,j₂ω_max = last(m)

		proc_id_min_Gsrc = get_processor_id_from_split_array(
							(ℓ_arr,1:Nν_Gfn),j₂ω_min,num_procs)
		proc_id_max_Gsrc = get_processor_id_from_split_array(
							(ℓ_arr,1:Nν_Gfn),j₂ω_max,num_procs)

		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_min_Gsrc:proc_id_max_Gsrc)

		proc_id_min_Gobs = get_processor_id_from_split_array(
							(ℓ_arr,1:Nν_Gfn),j₁ω_min,num_procs_obs)
		proc_id_max_Gobs = get_processor_id_from_split_array(
							(ℓ_arr,1:Nν_Gfn),j₁ω_max,num_procs_obs)

		Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,proc_id_min_Gobs:proc_id_max_Gobs)

		K = zeros(nr,nϕ)

		Gsrc = zeros(ComplexF64,nr,0:1)
		drGsrc = zeros(ComplexF64,nr,0:0)

		Gobs = zeros(ComplexF64,nr,0:1)
		drGobs = zeros(ComplexF64,nr,0:0)
		f_robs_rsrc = zeros(ComplexF64,nr)

		Pobs,Psrc = zeros(0:lmax,nϕ),zeros(0:lmax,nϕ)
		for (ϕ_ind,ϕ) in enumerate(ϕ_arr)
			Pobs[:,ϕ_ind] .= Pl(cosχ((θ,ϕ),xobs),lmax=lmax)
			Psrc[:,ϕ_ind] .= Pl(cosχ((θ,ϕ),xsrc),lmax=lmax)
		end

		Pobs,Psrc = permutedims.((Pobs,Psrc))

		for (ind,(j₁,j₂,ω_ind)) in enumerate(modes_iter_proc)

			ω = ω_arr[ω_ind]

			# Green function about the source radius
			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				(ℓ_arr,1:Nν_Gfn),(j₂,ω_ind),num_procs,:,1:2,1,1)

			# Derivative of Green function about the source radius
			read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
				(ℓ_arr,1:Nν_Gfn),(j₂,ω_ind),num_procs,:,1,1,2)

			# Green function about receiver location
			read_Gfn_file_at_index!(Gobs,Gfn_fits_files_obs,
				(ℓ_arr,1:Nν_Gfn),(j₁,ω_ind),num_procs_obs,:,1:2,1,1)

			# Derivative of Green function about receiver location
			read_Gfn_file_at_index!(drGobs,Gfn_fits_files_obs,
				(ℓ_arr,1:Nν_Gfn),(j₁,ω_ind),num_procs_obs,:,1,1,2)

			radial_fn_δc_firstborn!(f_robs_rsrc,Gsrc,drGsrc,j₂,
				Gobs,drGobs,j₁)

			for ϕ_ind in eachindex(ϕ_arr)
				@. K[:,ϕ_ind] +=  dω/2π * Powspec(ω) * 
							(2j₁+1)/4π * (2j₂+1)/4π * Pobs[ϕ_ind,j₁]*Psrc[ϕ_ind,j₂]*
							2real(conj(hω[ω_ind]) * f_robs_rsrc)
			end
		end

		map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs))

		return K
	end

	return pmapsum(summodes,modes_iter)
end

function Kc_longitudinal_slice_Mandal(xobs::Point3D,xsrc::Point3D;kwargs...)
	
	# Slice at a constant ϕ
	ϕ = get(kwargs,:ϕ,(xobs.ϕ + xsrc.ϕ)/2)

	Gfn_path_src,Gfn_path_obs = 
		Gfn_path_from_source_radius.((xsrc.r,xobs.r))

	@load(joinpath(Gfn_path_src,"parameters.jld2"),
		ω_arr,ℓ_arr,num_procs,dω,ν_start_zeros,Nν_Gfn)

	num_procs_obs = get_numprocs(Gfn_path_obs)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr); lmax = maximum(ℓ_range)
	ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
	modes_iter = Iterators.product(ℓ_range,ℓ_range,ν_ind_range)

	r_obs_ind = radial_grid_index(xobs.r)

	if haskey(kwargs,:hω)
		hω = kwargs.hω
	else
		hω = hω(xobs,xsrc;kwargs...)
	end
	hω = OffsetVector(hω[ν_start_zeros .+ ν_ind_range],ν_ind_range)

	nθ = get(kwargs,:nθ,lmax)
	θ_arr = get(kwargs,:θ,LinRange(0,π,nθ))
	nθ = length(θ_arr)

	function summodes(modes_iter_proc)

		m = moderanges_common_lastarray(modes_iter_proc)
		
		j₁ω_min,j₂ω_min = first(m)
		j₁ω_max,j₂ω_max = last(m)

		proc_id_min_Gsrc = get_processor_id_from_split_array(
							(ℓ_arr,1:Nν_Gfn),j₂ω_min,num_procs)
		proc_id_max_Gsrc = get_processor_id_from_split_array(
							(ℓ_arr,1:Nν_Gfn),j₂ω_max,num_procs)

		Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
								proc_id_min_Gsrc:proc_id_max_Gsrc)

		proc_id_min_Gobs = get_processor_id_from_split_array(
							(ℓ_arr,1:Nν_Gfn),j₁ω_min,num_procs_obs)
		proc_id_max_Gobs = get_processor_id_from_split_array(
							(ℓ_arr,1:Nν_Gfn),j₁ω_max,num_procs_obs)

		Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,
								proc_id_min_Gobs:proc_id_max_Gobs)

		K = zeros(nr,nθ)

		Gsrc = zeros(ComplexF64,nr,0:1)
		drGsrc = zeros(ComplexF64,nr,0:0)

		Gobs = zeros(ComplexF64,nr,0:1)
		drGobs = zeros(ComplexF64,nr,0:0)
		f_robs_rsrc = zeros(ComplexF64,nr)

		Pobs,Psrc = zeros(0:lmax,nϕ),zeros(0:lmax,nϕ)
		for (θ_ind,θ) in enumerate(θ_arr)
			Pobs[:,θ_ind] .= Pl(cosχ((θ,ϕ),xobs),lmax=lmax)
			Psrc[:,θ_ind] .= Pl(cosχ((θ,ϕ),xsrc),lmax=lmax)
		end

		Pobs,Psrc = permutedims.((Pobs,Psrc))

		for (ind,(j₁,j₂,ω_ind)) in enumerate(modes_iter_proc)

			ω = ω_arr[ω_ind]

			# Green function about the source radius
			read_Gfn_file_at_index!(Gsrc,Gfn_fits_files_src,
				(ℓ_arr,1:Nν_Gfn),(j₂,ω_ind),num_procs,:,1:2,1,1)

			# Derivative of Green function about the source radius
			read_Gfn_file_at_index!(drGsrc,Gfn_fits_files_src,
				(ℓ_arr,1:Nν_Gfn),(j₂,ω_ind),num_procs,:,1,1,2)
			
			# Green function about receiver location
			read_Gfn_file_at_index!(Gobs,Gfn_fits_files_obs,
				(ℓ_arr,1:Nν_Gfn),(j₁,ω_ind),num_procs_obs,:,1:2,1,1)

			# Derivative of Green function about receiver location
			read_Gfn_file_at_index!(drGobs,Gfn_fits_files_obs,
				(ℓ_arr,1:Nν_Gfn),(j₁,ω_ind),num_procs_obs,:,1,1,2)

			radial_fn_δc_firstborn!(f_robs_rsrc,Gsrc,drGsrc,j₂,
				Gobs,drGobs,j₁)

			@inbounds for θ_ind in eachindex(θ_arr), r_ind in eachindex(r)
				K[r_ind,θ_ind] +=  dω/2π * Powspec(ω) * 
							(2j₁+1)/4π * (2j₂+1)/4π * Pobs[θ_ind,j₁]*Psrc[θ_ind,j₂]*
							2real(conj(hω[ω_ind]) * f_robs_rsrc[r_ind])	
			end
		end

		map(closeGfnfits,(Gfn_fits_files_src,Gfn_fits_files_obs))

		return K
	end

	return pmapsum(summodes,modes_iter)
end

end # module