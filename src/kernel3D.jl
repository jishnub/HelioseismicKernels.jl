function populatekernel!(::Flow, K, Kₗₘ, Y, modes_lm, K_all_modes, Ω_ind, temp⁰, temp⁺¹)

	# The kernel is defined as
	# Kr = Σ_l K₀,ₗ₀*(r) Yₗ₀ + 2 Σ_{m=1}^{l} ℜ[K₀,ₗₘ* Yₗₘ]
	# Kθ = -√2 ℜ[ Σ_l K₁,ₗ₀*(r) Y¹ₗ₀ + Σ_{m=1}^{l} (K₁,ₗₘ* Y¹ₗₘ - K₋₁,ₗₘ* Y⁻¹ₗₘ) ]
	# Kϕ = √2 ℑ[ Σ_l K₁,ₗ₀*(r) Y¹ₗ₀ + Σ_{m=1}^{l} (K₁,ₗₘ* Y¹ₗₘ - K₋₁,ₗₘ* Y⁻¹ₗₘ) ]

	_modes_lm = SphericalHarmonicModes.ofordering(K_all_modes, modes_lm)
	for (lm_ind, (l,m)) in enumerate(_modes_lm)

		# Positive and negative m's are summed over, we don't need to loop over negative m's
		# separately
		m < 0 && continue

		Ylm = Y[(l,m)]

		@views begin

		if iszero(m)
			@. temp⁰ = real(conj(Kₗₘ[.., 0, lm_ind])*Ylm[0])
			@. temp⁺¹ = conj(Kₗₘ[.., 1, lm_ind])*Ylm[1]
		else
			# Sum of the +m and -m terms
			@. temp⁰ = 2real(conj(Kₗₘ[.., 0, lm_ind])*Ylm[0])
			@. temp⁺¹ = conj(Kₗₘ[.., 1, lm_ind])*Ylm[1] - Kₗₘ[..,-1, lm_ind]*conj(Ylm[-1])
		end

		end# views

		# r-component
		@. K[.., 1, Ω_ind] += temp⁰
		# θ-component
		@. K[.., 2, Ω_ind] += -√2*real(temp⁺¹)
		# ϕ-component
		@. K[.., 3, Ω_ind] += √2*imag(temp⁺¹)
	end
	return K
end

function Ku_longitudinal_slice_loop(localtimer, ϕθ::ProductSplit,
	Kₗₘ_fits_filename, modes_lm::LM, K_all_modes::LM)

	l_min, l_max = extrema(l_range(modes_lm))
	m_min, m_max = extrema(m_range(modes_lm))

	lm_ind_min = modeindex(K_all_modes, l_min, m_min)
	lm_ind_max = modeindex(K_all_modes, l_max, m_max)

	Kₗₘ = FITS(joinpath(SCRATCH_kerneldir[], Kₗₘ_fits_filename), "r") do f
		hdu = f[1]::ImageHDU{Float64,3}
		arr = reinterpret(ComplexF64, read(hdu))[.., lm_ind_min:lm_ind_max]
		OffsetArray(arr, :, -1:1, lm_ind_min:lm_ind_max)
	end

	K = zeros(size(Kₗₘ, 1), 3, length(ϕθ)) # 2nd axis is for vector components (r̂, ̂θ, ̂ϕ)

	temp⁺¹ = zeros(ComplexF64, nr)
	temp⁰ = zeros(nr)

	S = VectorSphericalHarmonics.cache(l_max)

	for (Ω_ind,(ϕ,θ)) in enumerate(ϕθ)
		VectorSphericalHarmonics.cache!(S, θ, ϕ)
		Y = genspharm(modes_lm, θ, ϕ, S)

		populatekernel!(Flow(), K, Kₗₘ, Y, modes_lm,
			K_all_modes, Ω_ind, temp⁰, temp⁺¹)
	end

	return K
end

function Ku_longitudinal_slice_from_Kₗₘ(comm, Kₗₘ_fits_filename; kwargs...)

	header = FITS(joinpath(SCRATCH_kerneldir[], Kₗₘ_fits_filename), "r") do f
		read_header(f[1])
	end

	K_all_modes = LM(header["L_MIN"]:header["L_MAX"],
					header["M_MIN"]:header["M_MAX"])

	l_min = get(kwargs,:l_min, header["L_MIN"]); header["L_MIN"] = l_min;
	l_max = get(kwargs,:l_max, header["L_MAX"]); header["L_MAX"] = l_max;
	m_min = get(kwargs,:m_min, min(header["M_MIN"], l_max)); header["M_MIN"] = m_min;
	m_max = get(kwargs,:m_max, min(header["M_MAX"], l_max)); header["M_MAX"] = m_max;
	kernel_modes = LM(l_min:l_max, m_min:m_max)
	nl = length(l_min:l_max)

	nθ_default = max(360, 4header["L_MAX"])
	nθ = get(kwargs,:nθ, nθ_default)
	θ_default = LinRange(0,π, nθ)
	θ = get(kwargs,:θ,θ_default)
	nθ = length(θ)
	ϕ = get(kwargs,:ϕ,(header["PHI1"] + header["PHI2"])/2)
	nl = length(l_range(kernel_modes))

	Krθ = pmapreduce(comm, Ku_longitudinal_slice_loop, (x...) -> cat(x..., dims = 3),
		(ϕ:ϕ,θ), Kₗₘ_fits_filename, kernel_modes, K_all_modes)

	Krθ === nothing && return nothing

	Krθ = permutedims(Krθ, [1, 3, 2]) # last axis will store the vector components

	# Write to fits file

	header["THSLMN"] = minimum(θ)
	set_comment!(header,"THSLMN","Minimum colatitude in slice")

	header["THSLMX"] = maximum(θ)
	set_comment!(header,"THSLMX","Maximum colatitude in slice")

	header["NTH"] = nθ
	set_comment!(header,"NTH","Number of grid points in colatitude")

	header["PHSL"] = ϕ
	set_comment!(header,"PHSL","Azimuth at which the slice is computed")

	FITS(joinpath(SCRATCH_kerneldir[],"Ku_rθslice_from_Klm.fits"),"w") do f
		FITSIO.write(f, Krθ, header=header)
	end
	Krθ
end

function Ku_latitudinal_slice_loop(localtimer, ϕθ::ProductSplit,
	Kₗₘ_fits_filename, modes_lm::LM, K_all_modes::LM)

	l_min, l_max = extrema(l_range(modes_lm))
	m_min, m_max = extrema(m_range(modes_lm))

	lm_ind_min = modeindex(K_all_modes, l_min, m_min)
	lm_ind_max = modeindex(K_all_modes, l_max, m_max)

	Kₗₘ = FITS(joinpath(SCRATCH_kerneldir[], Kₗₘ_fits_filename),"r") do f
		hdu = f[1]::ImageHDU{Float64,3}
		arr = reinterpret(ComplexF64, read(hdu))[.., lm_ind_min:lm_ind_max]
		OffsetArray(arr, axes(arr, 1), -1:1, lm_ind_min:lm_ind_max)
	end

	K = zeros(size(Kₗₘ, 1), 3, length(ϕθ)) # 2nd axis is for vector components (r,θ,ϕ)

	temp⁺¹ = zeros(ComplexF64, nr)
	temp⁰ = zeros(nr)

	S = VectorSphericalHarmonics.cache(l_max)

	for (Ω_ind,(ϕ,θ)) in enumerate(ϕθ)
		VectorSphericalHarmonics.cache!(S, θ, ϕ)
		Y = genspharm(modes_lm, θ, ϕ, S)

		populatekernel!(Flow(), K, Kₗₘ, Y, modes_lm,
			K_all_modes, Ω_ind, temp⁰, temp⁺¹)
	end

	return K
end

function Ku_latitudinal_slice_from_Kₗₘ(comm, Kₗₘ_fits_filename; kwargs...)
	header = FITS(joinpath(SCRATCH_kerneldir[], Kₗₘ_fits_filename), "r") do f
		read_header(f[1])
	end

	K_all_modes = LM(header["L_MIN"]:header["L_MAX"],
					header["M_MIN"]:header["M_MAX"])

	l_min = get(kwargs,:l_min, header["L_MIN"]); header["L_MIN"] = l_min;
	l_max = get(kwargs,:l_max, header["L_MAX"]); header["L_MAX"] = l_max;
	m_min = get(kwargs,:m_min, min(header["M_MIN"], l_max)); header["M_MIN"] = m_min;
	m_max = get(kwargs,:m_max, min(header["M_MAX"], l_max)); header["M_MAX"] = m_max;
	kernel_modes = LM(l_min:l_max, m_min:m_max)
	nl = length(l_min:l_max)

	θ = get(kwargs,:θ,(header["TH1"] + header["TH2"])/2)
	nϕ_default = max(720, 4header["L_MAX"])
	nϕ = get(kwargs,:nϕ, nϕ_default)
	ϕ_default = LinRange(0, 2π, nϕ)
	ϕ = get(kwargs,:ϕ,ϕ_default)
	nϕ = length(ϕ)
	nl = length(l_range(kernel_modes))

	Krϕ = pmapreduce(comm, Ku_latitudinal_slice_loop, (x...) -> cat(x..., dims = 3), (ϕ, θ:θ),
		Kₗₘ_fits_filename, kernel_modes, K_all_modes)

	Krϕ === nothing && return nothing

	Krϕ = permutedims(Krϕ,[1, 3, 2]) # last axis will store the vector components

	header["PHSLMN"] = minimum(ϕ)
	set_comment!(header,"PHSLMN","Minimum azimuth in slice")

	header["PHSLMX"] = maximum(ϕ)
	set_comment!(header,"PHSLMX","Maximum azimuth in slice")

	header["NPHI"] = nϕ
	set_comment!(header,"NPHI","Number of grid points in azimuth")

	header["THSL"] = θ
	set_comment!(header,"THSL","Colatitude at which the slice is computed")

	FITS(joinpath(SCRATCH_kerneldir[], "Ku_rϕslice_from_Klm.fits"),"w") do f
		FITSIO.write(f, Krϕ, header=header)
	end
	Krϕ
end

function Ku_surface_slice_loop(localtimer, ϕθ::ProductSplit,
	Kₗₘ_fits_filename, modes_lm::LM, K_all_modes::LM, r_obs_ind)

	l_min, l_max = extrema(l_range(modes_lm))
	m_min, m_max = extrema(m_range(modes_lm))

	lm_ind_min = modeindex(K_all_modes, l_min, m_min)
	lm_ind_max = modeindex(K_all_modes, l_max, m_max)

	Kₗₘ = FITS(joinpath(SCRATCH_kerneldir[], Kₗₘ_fits_filename),"r") do f
		hdu = f[1]::ImageHDU{Float64,3}
		arr = reinterpret_as_complex(read(hdu))[r_obs_ind, :, lm_ind_min:lm_ind_max]
		OffsetArray(arr, -1:1, lm_ind_min:lm_ind_max)
	end

	K = zeros(3, length(ϕθ))

	temp⁺¹ = zeros(ComplexF64)
	temp⁰ = zeros()

	V = VectorSphericalHarmonics.VSHCache(Float64, modes_lm)

	for (Ω_ind, (ϕ, θ)) in enumerate(ϕθ)
		Y = genspharm!(V, θ, ϕ)

		populatekernel!(Flow(), K, Kₗₘ, Y, modes_lm,
			K_all_modes, Ω_ind, temp⁰, temp⁺¹)
	end

	return K
end

function Ku_surface_slice_from_Kₗₘ(comm, Kₗₘ_fits_filename; kwargs...)

	r_obs = get(kwargs,:r_obs, r_obs_default)

	r_obs_ind = radial_grid_index(r_obs)

	header = FITS(joinpath(SCRATCH_kerneldir[], Kₗₘ_fits_filename), "r") do f
		read_header(f[1])
	end

	K_all_modes = LM(header["L_MIN"]:header["L_MAX"],
					header["M_MIN"]:header["M_MAX"])

	l_min = get(kwargs,:l_min, header["L_MIN"]); header["L_MIN"] = l_min;
	l_max = get(kwargs,:l_max, header["L_MAX"]); header["L_MAX"] = l_max;
	m_min = get(kwargs,:m_min, min(header["M_MIN"], l_max)); header["M_MIN"] = m_min;
	m_max = get(kwargs,:m_max, min(header["M_MAX"], l_max)); header["M_MAX"] = m_max;
	kernel_modes = LM(l_min:l_max, m_min:m_max)
	nl = length(l_min:l_max)

	nϕ_default = max(720, 4header["L_MAX"])
	nϕ = get(kwargs,:nϕ, nϕ_default)
	ϕ_default = LinRange(0, 2π, nϕ)
	ϕ = get(kwargs,:ϕ,ϕ_default)
	nϕ = length(ϕ)

	nθ_default = max(360, 4header["L_MAX"])
	nθ = get(kwargs,:nθ, nθ_default)
	θ_default = LinRange(0,π, nθ)
	θ = get(kwargs,:θ,θ_default)
	nθ = length(θ)
	nl = length(l_range(kernel_modes))

	K = pmapreduce(comm, Ku_surface_slice_loop, hcat, (ϕ,θ),
			Kₗₘ_fits_filename, kernel_modes, K_all_modes, r_obs_ind)

	K === nothing && return nothing

	Kϕθ = permutedims(reshape(K, 3, nϕ, nθ),[2, 3, 1]) # last axis will store the vector components

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

	FITS(joinpath(SCRATCH_kerneldir[], "Ku_ϕθ_from_Klm.fits"),"w") do f
		FITSIO.write(f, Kϕθ, header=header)
	end
	Kϕθ
end

#######################################################################

function populatekernel!(::SoundSpeed, K, Kₗₘ, Y, modes_lm, Ω_ind)

	# The kernel is defined as K = Σ_l Kₗ₀(r) Yₗ₀ + 2 Σ_{m=1}^{l} ℜ[Kₗₘ Yₗₘ]

	for (lm_ind,(l, m)) in enumerate(modes_lm)

		# don't need to iterate over negative m's as they are already summed over
		(m < 0) && continue

		# lm_ind = modeindex(modes_lm,(l, m))

		Ylm = Y[(l,m)]

		@views begin
			if iszero(m)
				# Term is purely real
				@. K[.., Ω_ind] += real(Kₗₘ[.., lm_ind])*real(Ylm)
			else
				# Sum of the +t and -t terms
				@. K[.., Ω_ind] += 2real(Kₗₘ[.., lm_ind]*Ylm)
			end
		end
	end
end

function Kc_angular_slice_loop(localtimer, ϕθ::ProductSplit, Klm_fits_filename, st_iterator::LM)

	l_max = maximum(l_range(st_iterator))

	S = SphericalHarmonics.cache(l_max)

	Klm = FITS(joinpath(SCRATCH_kerneldir[], Klm_fits_filename),"r") do f
		hdu = f[1]::ImageHDU{Float64,2}
		reinterpret(ComplexF64, read(hdu))
	end

	K = zeros(size(Klm, 1), length(ϕθ))

	for (Ω_ind,(ϕ,θ)) in enumerate(ϕθ)
		computePlmcostheta!(S, θ)
		Ylm = computeYlm!(S, θ, ϕ)
		populatekernel!(SoundSpeed(), K, Klm, Ylm, st_iterator, Ω_ind)
	end
	return K
end

function Kc_longitudinal_slice_from_Kₗₘ(comm, Kst_fits_filename; kwargs...)

	r_src = get(kwargs,:r_src, r_src_default)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)

	@load(joinpath(Gfn_path_src,"parameters.jld2"),ℓ_arr)

	header = FITS(joinpath(SCRATCH_kerneldir[], Kst_fits_filename), "r") do f
		read_header(f[1])
	end

	l_max = header["L_MAX"]
	m_max = header["M_MAX"]
	l_min = header["L_MIN"]
	m_min = header["M_MIN"]
	st_iterator = LM(l_min:l_max, m_min:m_max)

	θ = get(kwargs,:θ, LinRange(0,π, get(kwargs,:nθ, 4l_max)))
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

	Krθ = pmapreduce(comm, Kc_angular_slice_loop, hcat, (ϕ:ϕ, θ),
			Kst_fits_filename, st_iterator;
			progress_str = "Angles processed : ")

	Krθ === nothing && return nothing

	FITS(joinpath(SCRATCH_kerneldir[], "Krθ_from_Kst.fits"),"w") do f
		FITSIO.write(f, Krθ, header=header)
	end
	Krθ
end

function Kc_latitudinal_slice_from_Kₗₘ(comm, Kst_fits_filename; kwargs...)

	r_src = get(kwargs,:r_src, r_src_default)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)

	@load(joinpath(Gfn_path_src,"parameters.jld2"),ℓ_arr)

	header = FITS(joinpath(SCRATCH_kerneldir[], Kst_fits_filename),"r") do f
		read_header(f[1])
	end

	l_max = header["L_MAX"]
	m_max = header["M_MAX"]
	l_min = header["L_MIN"]
	m_min = header["M_MIN"]
	st_iterator = LM(l_min:l_max, m_min:m_max)

	ϕ = get(kwargs,:ϕ, LinRange(0, 2π, get(kwargs,:nϕ, 4l_max)))
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

	Krϕ = pmapreduce(comm, Kc_angular_slice_loop, hcat, (ϕ, θ:θ),
			Kst_fits_filename, st_iterator)

	Krϕ === nothing && return nothing
	FITS(joinpath(SCRATCH_kerneldir[],"Krϕ_from_Kst.fits"),"w") do f
		FITSIO.write(f, Krϕ, header=header)
	end
	Krϕ
end

function Kc_surface_slice_loop(localtimer, ϕθ::ProductSplit,
	Klm_fits_filename, modes_lm, r_obs_ind)

	l_max = maximum(l_range(modes_lm))

	S = SphericalHarmonics.cache(l_max)

	Klm = FITS(joinpath(SCRATCH_kerneldir[], Klm_fits_filename),"r") do f
		hdu = f[1]::ImageHDU{Float64,2}
		reinterpret(ComplexF64, read(hdu))[r_obs_ind,:]
	end

	K = zeros(length(ϕθ))

	for (Ω_ind,(ϕ,θ)) in enumerate(ϕθ)
		computePlmcostheta!(S, θ)
		Ylm = computeYlm!(S, θ, ϕ)

		populatekernel!(SoundSpeed(), K, Klm, Ylm, modes_lm,Ω_ind)
	end
	return K
end

function Kc_surface_slice_from_Kₗₘ(comm, Kst_fits_filename; kwargs...)

	r_src = get(kwargs,:r_src, r_src_default)
	r_obs = get(kwargs,:r_obs, r_obs_default)

	r_obs_ind = radial_grid_index(r_obs)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)

	@load(joinpath(Gfn_path_src,"parameters.jld2"),ℓ_arr)

	header = FITS(joinpath(SCRATCH_kerneldir[], Kst_fits_filename),"r") do f
		read_header(f[1])
	end

	l_max = header["L_MAX"]
	m_max = header["M_MAX"]
	l_min = header["L_MIN"]
	m_min = header["M_MIN"]
	st_iterator = LM(l_min:l_max, m_min:m_max)

	ϕ = get(kwargs,:ϕ, LinRange(0, 2π, get(kwargs,:nϕ, 4l_max)))
	nϕ = length(ϕ)
	θ = get(kwargs,:θ, LinRange(0, π, get(kwargs,:nθ, 4l_max)))
	nθ = length(θ)

	K = pmapreduce(comm, Kc_surface_slice_loop, vcat, (ϕ,θ),
			Kst_fits_filename, st_iterator, r_obs_ind)

	K === nothing && return nothing

	Kϕθ = reshape(K, nϕ, nθ)

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

	FITS(joinpath(SCRATCH_kerneldir[],"Kϕθ_from_Kst.fits"),"w") do f
		FITSIO.write(f, Kϕθ, header=header)
	end
	Kϕθ
end

#######################################################################
# The following function compute the 3D kernel using the fleshed-out formulation
# and not from the spherical harmonic coefficients
# They are only defined for radial displacements, and are used to compare
# our results with those obtained using a formalism similar to Mandal et al. (2017)
#######################################################################

function populatekernel3D!(::SoundSpeed, K,Ω_ind,
	realconjhωHjₒjₛω_r₁r₂,     Pjₒ1_Pjₛ2,
	realconjhωconjHjₒjₛω_r₂r₁, Pjₒ2_Pjₛ1,
	pre)

	P12 = Pjₒ1_Pjₛ2[Ω_ind]
	P21 = Pjₒ2_Pjₛ1[Ω_ind]

	iszero(P12) && iszero(P21) && return

	for r_ind in UnitRange(axes(K, 1))
		K[r_ind, Ω_ind] += pre * (
						realconjhωHjₒjₛω_r₁r₂[r_ind] * P12  +
						realconjhωconjHjₒjₛω_r₂r₁[r_ind] * P21 )
	end
end

function Kc_partial(localtimer, modes_iter_proc::ProductSplit,
	xobs1, xobs2, los::los_radial, θ_arr, ϕ_arr, hω_arr,
	p_Gsrc::Union{Nothing, ParamsGfn}=nothing,
	p_Gobs1::Union{Nothing, ParamsGfn}=nothing,
	p_Gobs2::Union{Nothing, ParamsGfn}=p_Gobs1,
	r_src=r_src_default, r_obs=nothing)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src=r_src)
	p_Gobs1 = read_all_parameters(p_Gobs1, r_src=xobs1.r)
	p_Gobs2 = read_all_parameters(p_Gobs2, r_src=xobs2.r)

	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	Gfn_path_obs1, NGfn_files_obs1 =  p_Gobs1.path, p_Gobs1.num_procs
	Gfn_path_obs2, NGfn_files_obs2 = p_Gobs2.path, p_Gobs2.num_procs
	@unpack ℓ_arr,ω_arr, Nν_Gfn, dω = p_Gsrc

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))
	obs_at_same_height = r₁_ind == r₂_ind

	nθ, nϕ = length(θ_arr), length(ϕ_arr)
	nΩ = nθ * nϕ
	θϕ_iter = Iterators.product(θ_arr,ϕ_arr)

	# jₛ is the angular degree of the wave from the source, and
	# jₒ is the angular degree of the wave from the observation point
	jₛ_min_proc, jₛ_max_proc = extrema(modes_iter_proc, 1)
	νind_min_proc,νind_max_proc = extrema(modes_iter_proc, 2)
	j_min, j_max = extrema(ℓ_arr)
	jₛ_range_proc = jₛ_min_proc:jₛ_max_proc
	jₒjₛ_allmodes = L₂L₁Δ(jₛ_range_proc, 2j_max, j_min:j_max)
	jₒ_range_proc = l₂_range(jₒjₛ_allmodes)

	# Need to load all the modes on this processor
	modes_obsGfn_proc = ProductSplit((jₒ_range_proc,νind_min_proc:νind_max_proc), 1, 1)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr, 1:Nν_Gfn), modes_iter_proc, NGfn_files_src)

	# Gℓ′ω(r, robs) files
	Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_obs1,
						(ℓ_arr, 1:Nν_Gfn), modes_obsGfn_proc, NGfn_files_obs1)

	Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_obs2,
						(ℓ_arr, 1:Nν_Gfn), modes_obsGfn_proc, NGfn_files_obs2)

	K = zeros(nr, nΩ)

	arrs = allocatearrays(SoundSpeed(), los, obs_at_same_height)
	@unpack Gsrc, drGsrc, Gobs1, drGobs1, Gobs2, drGobs2, divGsrc, divGobs = arrs
	@unpack fjₒjₛ_r₁_rsrc, fjₒjₛ_r₂_rsrc, Hjₒjₛω_r₁r₂, Hjₒjₛω_r₂r₁ = arrs
	@unpack Gobs1_cache, drGobs1_cache, Gobs2_cache, drGobs2_cache = arrs
	@unpack tworealconjhωHjₒjₛω_r₁r₂, tworealconjhωconjHjₒjₛω_r₂r₁ = arrs

	@unpack Gobs1_cache, Gobs2_cache = arrs
	for jₒ in jₒ_range_proc
		Gobs1_cache[jₒ] = zeros(ComplexF64, axes(Gsrc))
		drGobs1_cache[jₒ] = zeros(ComplexF64, axes(drGsrc))
		if !obs_at_same_height
			Gobs2_cache[jₒ] = zeros(ComplexF64, axes(Gsrc))
			drGobs2_cache[jₒ] = zeros(ComplexF64, axes(drGsrc))
		end
	end

	@timeit localtimer "legendre" begin
	P1, P2 = zeros(0:j_max, nΩ), zeros(0:j_max, nΩ)
	for (Ω_ind,(θ,ϕ)) in enumerate(θϕ_iter)
		collectPl!(view(P1, :, Ω_ind), cosχ((θ,ϕ), xobs1), lmax = j_max)
		collectPl!(view(P2, :, Ω_ind), cosχ((θ,ϕ), xobs2), lmax = j_max)
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
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn),(jₛ,ω_ind), NGfn_files_src,:, 1:2, 1, 1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn),(jₛ,ω_ind), NGfn_files_src,:, 1:1, 1, 2)

		end # timer

		Grr_r₁_rsrc = Gsrc[r₁_ind, 0]
		Grr_r₂_rsrc = Gsrc[r₂_ind, 0]

		jₒ_range_jₛ = l₂_range(jₒjₛ_allmodes, jₛ)

		for jₒ in jₒ_range_jₛ

			pre = dωω²Pω * (2jₒ+1)/4π * (2jₛ+1)/4π

			Gobs1 = Gobs1_cache[jₒ]
			drGobs1 = drGobs1_cache[jₒ]
			Gobs2 = Gobs2_cache[jₒ]
			drGobs2 = drGobs2_cache[jₒ]

			@timeit localtimer "FITS" begin

			# Green function about receiver location
			if ω_ind != ω_ind_prev || jₒ ∉ l₂_range(jₒjₛ_allmodes, jₛ-1)
				read_Gfn_file_at_index!(Gobs1, Gfn_fits_files_obs1,
					(ℓ_arr, 1:Nν_Gfn),(jₒ,ω_ind), NGfn_files_obs1,:, 1:2, 1, 1)

				# Derivative of Green function about receiver location
				read_Gfn_file_at_index!(drGobs1, Gfn_fits_files_obs1,
					(ℓ_arr, 1:Nν_Gfn),(jₒ,ω_ind), NGfn_files_obs1,:, 1:1, 1, 2)
			end
			end # timer

			@timeit localtimer "radial term" begin
			# precompute the radial term in f
			radial_fn_δc_firstborn!(fjₒjₛ_r₁_rsrc, Gsrc, drGsrc, jₛ, divGsrc,
				Gobs1, drGobs1, jₒ, divGobs)
			end # timer

			@timeit localtimer "radial term 2" begin
			Hjₒjₛω!(Hjₒjₛω_r₁r₂, fjₒjₛ_r₁_rsrc, Grr_r₂_rsrc)
			end
			@timeit localtimer "radial term 3" begin
			@. tworealconjhωHjₒjₛω_r₁r₂ = 2real(conjhω * Hjₒjₛω_r₁r₂)
			end

			if r₁_ind != r₂_ind
				@timeit localtimer "FITS" begin
				if ω_ind != ω_ind_prev || jₒ ∉ l₂_range(jₒjₛ_allmodes, jₛ-1)
					# Green function about receiver location
					read_Gfn_file_at_index!(Gobs2, Gfn_fits_files_obs2,
						(ℓ_arr, 1:Nν_Gfn),(jₒ,ω_ind), NGfn_files_obs2,:, 1:2, 1, 1)

					# Derivative of Green function about receiver location
					read_Gfn_file_at_index!(drGobs2, Gfn_fits_files_obs2,
						(ℓ_arr, 1:Nν_Gfn),(jₒ,ω_ind), NGfn_files_obs2,:, 1:1, 1, 2)
				end
				end # timer

				@timeit localtimer "radial term" begin
				radial_fn_δc_firstborn!(fjₒjₛ_r₂_rsrc, Gsrc, drGsrc, jₛ, divGsrc,
					Gobs2, drGobs2, jₒ, divGobs)

				end # timer
			end
			@timeit localtimer "radial term 2" begin
			Hjₒjₛω!(Hjₒjₛω_r₂r₁, fjₒjₛ_r₂_rsrc, Grr_r₁_rsrc)
			end
			@timeit localtimer "radial term 3" begin
			@. tworealconjhωconjHjₒjₛω_r₂r₁ = 2real(conjhω * conj(Hjₒjₛω_r₂r₁))
			end

			@timeit localtimer "legendre" begin
			@inbounds for Ωind in eachindex(Pjₒ1_Pjₛ2)
				Pjₒ1_Pjₛ2[Ωind] = P1[Ωind, jₒ]*P2[Ωind, jₛ]
				Pjₒ2_Pjₛ1[Ωind] = P1[Ωind, jₛ]*P2[Ωind, jₒ]
			end
			end # timer

			@timeit localtimer "kernel" begin
			for Ω_ind in axes(K, 2)
				populatekernel3D!(SoundSpeed(), K, Ω_ind,
				tworealconjhωHjₒjₛω_r₁r₂, Pjₒ1_Pjₛ2,
				tworealconjhωconjHjₒjₛω_r₂r₁, Pjₒ2_Pjₛ1,
				pre)
			end
			end # timer
		end

		ω_ind_prev = ω_ind


	end

	map(closeGfnfits,(Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2))

	signaltomaster!(timers_channel, localtimer)
	map(finalize_except_wherewhence,(progress_channel, timers_channel))

	return K
end

function generatefitsheader()
    FITSHeader(["r1","th1","phi1","r2","th2","phi2",
		"l_max","m_max","jmin","jmax","nuimin","nuimax",
		"PHSLMN","PHSLMX","NPHI",
		"THSLMN","THSLMX","NTH"],
		Any[float(xobs1.r), float(xobs1.θ), float(xobs1.ϕ),
		float(xobs2.r), float(xobs2.θ), float(xobs2.ϕ), Int(l_max), Int(m_max),
		minimum(ℓ_range), maximum(ℓ_range), minimum(ν_ind_range), maximum(ν_ind_range),
		minimum(ϕ), maximum(ϕ), length(ϕ),
		minimum(θ), maximum(θ), length(θ)],
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

function _Kc(comm, xobs1::Point3D, xobs2::Point3D, los::los_direction;θ,ϕ, kwargs...)
	r_src, r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc, p_Gobs1, p_Gobs2 = read_parameters_for_points(xobs1, xobs2; kwargs...)

	@unpack ν_arr,ℓ_arr = p_Gsrc
	ℓ_range, ν_ind_range, modes_iter = ℓ_and_ν_range(ℓ_arr,ν_arr; kwargs...)

	hω_arr = get(kwargs,:hω) do
		h = hω(comm, TravelTimes(), xobs1, xobs2, los; kwargs...,
			ℓ_range=ℓ_range, ν_ind_range=ν_ind_range,
			print_timings=false)
		_broadcast(h, 0, comm)
	end

	pmapsum_timed(comm, Kc_partial, modes_iter,
		xobs1, xobs2, los,θ,ϕ, hω_arr,
		p_Gsrc, p_Gobs1, p_Gobs2, r_src, r_obs)
end

function Kc_latitudinal_slice(comm, xobs1::Point3D, xobs2::Point3D, los::los_radial=los_radial(); kwargs...)

	p_Gsrc = read_all_parameters(; kwargs...)
	@unpack ℓ_arr,ν_arr = p_Gsrc
	ℓ_range, = ℓ_and_ν_range(ℓ_arr,ν_arr; kwargs...)
	_,ℓmax = extrema(ℓ_range)

	ϕ = get(kwargs,:ϕ, LinRange(0, 2π, get(kwargs,:nϕ, 4ℓmax)))
	θ = get(kwargs,:θ,(xobs1.θ+xobs2.θ)/2)

	Kc_3D_lat_slice = _Kc(comm, xobs1, xobs2, los; kwargs..., θ=θ, ϕ=ϕ)
	Kc_3D_lat_slice === nothing && return nothing

	header = generatefitsheader()

	filename = joinpath(SCRATCH_kerneldir[],
		"Kc_3D_lat_slice_jcutoff$(ℓmax).fits")

	FITS(filename,"w") do f
		FITSIO.write(f, Kc_3D_lat_slice, header=header)
	end
	Kc_3D_lat_slice
end

function Kc_longitudinal_slice(comm, xobs1::Point3D, xobs2::Point3D, los::los_radial=los_radial(); kwargs...)
	p_Gsrc = read_all_parameters(; kwargs...)
	@unpack ℓ_arr,ν_arr = p_Gsrc
	ℓ_range, = ℓ_and_ν_range(ℓ_arr,ν_arr; kwargs...)
	_,ℓmax = extrema(ℓ_range)

	θ = get(kwargs,:θ, LinRange(0,π, get(kwargs,:nθ, 4ℓmax)))
	ϕ = get(kwargs,:ϕ,(xobs1.ϕ+xobs2.ϕ)/2)

	Kc_3D_long_slice = _Kc(comm, xobs1, xobs2, los; kwargs..., θ=θ, ϕ=ϕ)
	Kc_3D_long_slice === nothing && return nothing

	header = generatefitsheader()

	filename = joinpath(SCRATCH_kerneldir[],
		"Kc_3D_long_slice_jcutoff$(ℓmax).fits")

	FITS(filename,"w") do f
		FITSIO.write(f, Kc_3D_long_slice, header=header)
	end
	Kc_3D_long_slice
end

# 3D profile
function Kc(comm, xobs1::Point3D, xobs2::Point3D, los::los_radial=los_radial(); kwargs...)

	p_Gsrc = read_all_parameters(; kwargs...)
	@unpack ℓ_arr,ν_arr = p_Gsrc
	ℓ_range,_ = ℓ_and_ν_range(ℓ_arr,ν_arr; kwargs...)
	_,ℓmax = extrema(ℓ_range)
	# These may be derived form ℓmax

	l_max = get(kwargs,:l_max, 2ℓmax);

	Nnodes = get(kwargs,:nθ, ceil(Int,(l_max+1)/2))
	nodes,_ = gausslegendre(Nnodes)
	θdefault = acos.(nodes)[end:-1:1] # flipped to get to increasing θ from increasing cosθ

	θ = get(kwargs,:θ,θdefault)
	ϕ = get(kwargs,:ϕ, LinRange(0, 2π, get(kwargs,:nϕ, 2l_max+1)))

	K = _Kc(comm, xobs1, xobs2, los; kwargs...,θ=θ,ϕ=ϕ)
	K === nothing && return nothing
	Kc_3D = copy(reshape(K, nr, length(θ), length(ϕ)))

	header = generatefitsheader()

	filename = joinpath(SCRATCH_kerneldir[],
		"Kc_3D_jcutoff$(ℓmax).fits")

	FITS(filename,"w") do f
		FITSIO.write(f, Kc_3D, header=header)
	end
	Kc_3D
end
