l_min(SHModes::SphericalHarmonicModes.SHModeRange) = minimum(l_range(SHModes))
l_max(SHModes::SphericalHarmonicModes.SHModeRange) = maximum(l_range(SHModes))

m_min(SHModes::SphericalHarmonicModes.SHModeRange) = minimum(m_range(SHModes))
m_max(SHModes::SphericalHarmonicModes.SHModeRange) = maximum(m_range(SHModes))

########################################################################################
# Utility function
########################################################################################

function getkernelmodes(; kwargs...)
	s_min = get(kwargs, :s_min, 0) :: Int
	s_max = get(kwargs, :s_max, s_min) :: Int
	t_min = get(kwargs, :t_min, 0) :: Int
	t_max = get(kwargs, :t_max, t_min) :: Int

	m = get(kwargs, :SHModes, LM(s_min:s_max, t_min:t_max))
	LM{UnitRange{Int}, UnitRange{Int}}(m)
end

function unpackGfnparams(p_Gsrc, r_src, p_Gobs1, r_obs1, p_Gobs2, r_obs2)
	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src)
	p_Gobs1 = read_all_parameters(p_Gobs1, r_src = r_obs1)
	p_Gobs2 = read_all_parameters(p_Gobs2, r_src = r_obs2)

	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	Gfn_path_obs1, NGfn_files_obs1 =  p_Gobs1.path, p_Gobs1.num_procs
	Gfn_path_obs2, NGfn_files_obs2 = p_Gobs2.path, p_Gobs2.num_procs

	(; Gfn_path_src, NGfn_files_src,
		Gfn_path_obs1, NGfn_files_obs1,
		Gfn_path_obs2, NGfn_files_obs2,
		p_Gsrc, p_Gobs1, p_Gobs2)
end

VSHType(::los_radial) = SH()
VSHType(::los_earth) = GSH()

function los_projected_spheroidal_biposh_flippoints(xobs1::SphericalPoint, xobs2::SphericalPoint, los, SHModes, jₒjₛ_allmodes)
	l1, l2 = line_of_sight_covariant.((xobs1, xobs2), los)
	_Y12, _Y21 = biposh_flippoints(VSHType(los), Point2D(xobs1)..., Point2D(xobs2)..., SHModes, jₒjₛ_allmodes)
	Y12 = biposh_spheroidal(los_projected_biposh!(los, _Y12, l1, l2))
	Y21 = biposh_spheroidal(los_projected_biposh!(los, _Y21, l2, l1))
	return Y12, Y21
end

function trim_m(Y::SHArray{<:Any,1, <:Vector, <:Tuple{LM}}, SHModes)
	Yp = parent(Y)
	modes = only(SphericalHarmonicArrays.modes(Y))
	modes_new = LM(l_range(modes), intersect(m_range(SHModes), m_range(modes)))
	inds = [ind for (ind,(l,m)) in enumerate(modes) if m in m_range(SHModes, l)]
	Yp = Yp[inds]
	SHArray(Yp, modes_new)
end
function trim_m(Y, SHModes)
	SHArray([trim_m(Y[ind], SHModes) for ind in eachindex(Y)],
		SphericalHarmonicArrays.modes(Y))
end

function rotate_biposh!(Y1′2′_j1j2, Y12_j1j2, M, D)
	modes = only(SphericalHarmonicArrays.modes(Y1′2′_j1j2))
	modes_allm = only(SphericalHarmonicArrays.modes(Y12_j1j2))
	for l in l_range(modes)
		Dl = D[l]
		for m in m_range(modes, l)
			Y1′2′_j1j2[(l,m)] = M * sum(Dl[m′,m] * Y12_j1j2[(l,m′)] for m′ in m_range(modes_allm, l))
		end
	end
	return Y1′2′_j1j2
end
function rotate_biposh(Y12, M, D, SHModes)
	Y1′2′ = trim_m(Y12, SHModes)
	for ind in eachindex(Y12)
		Y12_j1j2 = Y12[ind]
		Y1′2′_j1j2 = Y1′2′[ind]
		rotate_biposh!(Y1′2′_j1j2, Y12_j1j2, M, D)
	end
	return Y1′2′
end

function los_projected_spheroidal_biposh_flippoints((xobs1, xobs1′)::NTuple{2,SphericalPoint},
	(xobs2, xobs2′)::NTuple{2,SphericalPoint}, los, SHModes, jₒjₛ_allmodes)

	SHModes_allm = LM(l_range(SHModes))

	_Y12_allm, _Y21_allm = biposh_flippoints(VSHType(los), Point2D(xobs1)..., Point2D(xobs2)..., SHModes_allm, jₒjₛ_allmodes)
	_Y12 = trim_m(_Y12_allm, SHModes)
	_Y21 = trim_m(_Y21_allm, SHModes)
	Y12 = los_projected_biposh_spheroidal(_Y12, xobs1, xobs2, los)
	Y21 = los_projected_biposh_spheroidal(_Y21, xobs2, xobs1, los)
	R, M11′, M22′ = rotation_points_12(HelicityCovariant(), xobs1, xobs2, xobs1′, xobs2′)
	M11′22′ = Diagonal(LinearAlgebra.kron(M11′, M22′))
	M22′11′ = Diagonal(LinearAlgebra.kron(M22′, M11′))
	Rinv = inv(R)
	α, β, γ = Rinv.theta1, Rinv.theta2, Rinv.theta3
	lmax = maximum(l_range(SHModes))
	D = OffsetArray([OffsetArray(wignerD(j, α, β, γ), -j:j, -j:j) for j in 0:lmax], 0:lmax)
	_Y1′2′ = rotate_biposh(_Y12_allm, M11′22′, D, SHModes)
	_Y2′1′ = rotate_biposh(_Y21_allm, M22′11′, D, SHModes)
	Y1′2′ = los_projected_biposh_spheroidal(_Y1′2′, xobs1′, xobs2′, los)
	Y2′1′ = los_projected_biposh_spheroidal(_Y2′1′, xobs2′, xobs1′, los)
	return Y12, Y21, Y1′2′, Y2′1′
end

########################################################################################
# Validation for uniform rotation
########################################################################################

function populatekernelvalidation!(::Flow, ::los_radial, K::AbstractVector{<:Real},
	(j, ω), ∂ϕ₂Pⱼ_cosχ, conjhω, H¹₁jj_r₁r₂, H¹₁jj_r₂r₁, dω)

	pre = 2*√(3/π) * dω/2π * ω^3 * Powspec(ω) * ∂ϕ₂Pⱼ_cosχ
	for r_ind in UnitRange(axes(K, 1))
		K[r_ind] += pre * imag( conjhω * ( H¹₁jj_r₁r₂[r_ind] + conj(H¹₁jj_r₂r₁[r_ind]) ) )
	end
end

function populatekernelvalidation!(::Flow, ::los_earth, K::AbstractVector{<:Real},
	(j, ω), Y12j, conjhω, H¹₁jj_r₁r₂, H¹₁jj_r₂r₁, dω)

	pre = 2 * dω/2π * ω^3 * Powspec(ω) * √((j*(j+1))/π)

	for r_ind in UnitRange(axes(K, 1))
		for α₂ in 0:1, α₁ in 0:1
			Pʲʲ₁₀_α₁α₂ = pre * imag(Y12j[α₁, α₂])
			iszero(Pʲʲ₁₀_α₁α₂) && continue

			K[r_ind] += Pʲʲ₁₀_α₁α₂ * imag(conjhω *
				( H¹₁jj_r₁r₂[α₁, α₂, r_ind] + conj(H¹₁jj_r₂r₁[α₂, α₁, r_ind] ) ) )
		end
	end
end

function populatekernelvalidation!(::Flow, ::los_radial, K::AbstractMatrix{<:Real},
	(j, ω), ∂ϕ₂Pⱼ_cosχ::AbstractVector{<:Real},
	conjhω::AbstractVector{ComplexF64}, H¹₁jj_r₁r₂, H¹₁jj_r₂r₁, dω)

	pre = 2*√(3/4π) * dω/2π * ω^3 * Powspec(ω)
	for n2ind in axes(K, 2)
		pre2 = pre * ∂ϕ₂Pⱼ_cosχ[n2ind]
		conjhω_n2 = conjhω[n2ind]
		for r_ind in axes(K, 1)
			K[r_ind, n2ind] +=  pre2 * 2imag( conjhω_n2 * ( H¹₁jj_r₁r₂[r_ind] + conj(H¹₁jj_r₂r₁[r_ind]) ) )
		end
	end
end

function populatekernelvalidation!(::Flow, ::los_earth, K::AbstractMatrix{<:Real},
	(j, ω), Y12j::AbstractVector, conjhω, H¹₁jj_r₁r₂, H¹₁jj_r₂r₁, dω)

	pre = 2 * dω/2π * ω^3 * Powspec(ω) * √((j*(j+1))/π)

	for n2ind in axes(K, 2)
		Y12j_n₂ = Y12j[n2ind]
		conjhω_n₂ = conjhω[n2ind]

		for r_ind in UnitRange(axes(K, 1))
			for α₂ in 0:1, α₁ in 0:1
				Pʲʲ₁₀_α₁α₂ = pre * imag(Y12j_n₂[α₁, α₂])
				iszero(Pʲʲ₁₀_α₁α₂) && continue

				K[r_ind, n2ind] += Pʲʲ₁₀_α₁α₂ * imag(conjhω_n₂ *
					(H¹₁jj_r₁r₂[α₁, α₂, r_ind] + conj(H¹₁jj_r₂r₁[α₂, α₁, r_ind] ) ) )
			end
		end
	end
end

function kernel_ℑu⁺₁₀_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::SphericalPoint, xobs2::SphericalPoint, los::los_direction, hω,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default)

	@unpack p_Gsrc, p_Gobs1, p_Gobs2, Gfn_path_src, NGfn_files_src,
	Gfn_path_obs1, NGfn_files_obs1, Gfn_path_obs2, NGfn_files_obs2 =
		unpackGfnparams(p_Gsrc, r_src, p_Gobs1, radius(xobs1), p_Gobs2, radius(xobs2))

	@unpack ℓ_arr, ω_arr, Nν_Gfn, dω = p_Gsrc

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2 =
			Gfn_fits_files.((Gfn_path_src, Gfn_path_obs1, Gfn_path_obs2),
				((ℓ_arr, 1:Nν_Gfn),), (ℓ_ωind_iter_on_proc,),
				(NGfn_files_src, NGfn_files_obs1, NGfn_files_obs2))

	K = zeros(nr)

	arrs = allocatearrays(Flow(), los, r₁_ind == r₂_ind)
	@unpack Gsrc, Gobs1, Gobs2, G¹₁jj_r₁, G¹₁jj_r₂, H¹₁jj_r₁r₂, H¹₁jj_r₂r₁ = arrs

	Y12 = los_projected_biposh_spheroidal(computeY₁₀, xobs1, xobs2, los, ℓ_ωind_iter_on_proc)

	for (j, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]
		conjhω = conj(hω[ω_ind])

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (j, ω_ind), NGfn_files_src, 1:2, srcindFITS(los), :, 1)

		Gγr_r₁_rsrc = αrcomp(Gsrc, r₁_ind, los)
		Gγr_r₂_rsrc = αrcomp(Gsrc, r₂_ind, los)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1, Gfn_fits_files_obs1,
			(ℓ_arr, 1:Nν_Gfn), (j, ω_ind), NGfn_files_obs1, 1:2, srcindFITS(los), :, 1)

		radial_fn_uniform_rotation_firstborn!(G¹₁jj_r₁, Gsrc, Gobs1, j, los)
		Hjₒjₛω!(H¹₁jj_r₁r₂, G¹₁jj_r₁, Gγr_r₂_rsrc)

		if r₁_ind != r₂_ind
			# Green function about receiver location
			read_Gfn_file_at_index!(Gobs2, Gfn_fits_files_obs2,
			(ℓ_arr, 1:Nν_Gfn), (j, ω_ind), NGfn_files_obs2, 1:2, srcindFITS(los), :, 1)

			radial_fn_uniform_rotation_firstborn!(G¹₁jj_r₂, Gsrc, Gobs2, j, los)
			Hjₒjₛω!(H¹₁jj_r₂r₁, G¹₁jj_r₂, Gγr_r₁_rsrc)
		end

		populatekernelvalidation!(Flow(), los, K, (j, ω),
			Y12[j], conjhω, H¹₁jj_r₁r₂, H¹₁jj_r₂r₁, dω)
	end

	foreach(closeGfnfits, (Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2))
	return @. (ρ/r)*K
end

function kernel_ℑu⁺₁₀_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	nobs1::Point2D, nobs2_arr::Vector{<:Point2D}, los::los_direction, hω,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = p_Gobs1,
	r_src = r_src_default)

	hω = permutedims(hω) #  convert to (n2, ω)

	r_obs_ind = radial_grid_index(nobs1)

	@unpack p_Gsrc, p_Gobs1, Gfn_path_src, NGfn_files_src =
		unpackGfnparams(p_Gsrc, r_src, p_Gobs1, r_obs_default, p_Gobs2, r_obs_default)

	Gfn_path_obs, NGfn_files_obs = p_Gobs1.path, p_Gobs1.num_procs

	@unpack ℓ_arr, ω_arr, Nν_Gfn, dω = p_Gsrc

	Gfn_fits_files_src, Gfn_fits_files_obs =
			Gfn_fits_files.((Gfn_path_src, Gfn_path_obs),
				((ℓ_arr, 1:Nν_Gfn),), (ℓ_ωind_iter_on_proc,),
				(NGfn_files_src, NGfn_files_obs))

	K = zeros(nr, length(nobs2_arr))

	arrs = allocatearrays(Flow(), los, true)
	@unpack Gsrc, H¹₁jj_r₁r₂, G¹₁jj_r₁, Gobs1 = arrs

	Y12 = los_projected_biposh_spheroidal(computeY₁₀, nobs1, nobs2_arr, los, ℓ_ωind_iter_on_proc)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]
		conjhω = conj(@view hω[:, ω_ind])

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, 1:2, srcindFITS(los), :, 1)

		G_r₂_rsrc = αrcomp(Gsrc, r_obs_ind, los)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1, Gfn_fits_files_obs,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs, 1:2, srcindFITS(los), :, 1)

		radial_fn_uniform_rotation_firstborn!(G¹₁jj_r₁, Gsrc, Gobs1, ℓ, los)
		Hjₒjₛω!(H¹₁jj_r₁r₂, G¹₁jj_r₁, G_r₂_rsrc)

		populatekernelvalidation!(Flow(), los, K, (ℓ, ω),
			view(Y12, :, ℓ), conjhω, H¹₁jj_r₁r₂, H¹₁jj_r₁r₂, dω)
	end

	map(closeGfnfits, (Gfn_fits_files_src, Gfn_fits_files_obs))
	return @. (ρ/r)*K
end

# Traveltimes

kernelfilenameℑu⁺₁₀(::TravelTimes, ::los_radial) = "K_δτ_ℑu⁺₁₀.fits"
kernelfilenameℑu⁺₁₀(::TravelTimes, ::los_earth) = "K_δτ_ℑu⁺₁₀_los.fits"
kernelfilenameℑu⁺₁₀(::Amplitudes, ::los_radial) = "K_A_ℑu⁺₁₀.fits"
kernelfilenameℑu⁺₁₀(::Amplitudes, ::los_earth) = "K_A_ℑu⁺₁₀_los.fits"

function kernel_ℑu⁺₁₀(comm, m::SeismicMeasurement, xobs1, xobs2, los::los_direction = los_radial(); kwargs...)
	r_src, = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc, p_Gobs1, p_Gobs2 = read_parameters_for_points(xobs1, xobs2; kwargs...)
	@unpack ν_arr, ℓ_arr = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)

	# Compute the appropriate window function depending on the parameter of interest
	hω_arr = get(kwargs, :hω) do
		h = hω(comm, m, xobs1, xobs2, los; kwargs..., print_timings = false)
		_broadcast(h, 0, comm)
	end

	# Use the window function to compute the kernel
	K = pmapsum(comm, kernel_ℑu⁺₁₀_partial, iters,
		xobs1, xobs2, los, hω_arr, p_Gsrc, p_Gobs1, p_Gobs2, r_src)

	K === nothing && return nothing

	if get(kwargs, :save, true)
		filename = joinpath(SCRATCH_kerneldir[], kernelfilenameℑu⁺₁₀(m, los))
		FITS(filename,"w") do f
			write(f, reinterpret_as_float(K))
		end
	end

	return K
end

########################################################################################
# Kernels for flow velocity
########################################################################################

@inline function mulprefactor!(K::AbstractArray{<:Real}, vindskip = 0)
	@turbo @. K *= ρ
	@assert axes(K,1) == eachindex(r) "first axis of K does not match that of r"
	for lmind in UnitRange(axes(K, 3)), vind in UnitRange(axes(K, 2))
		vind == vindskip && continue # radial component
		@turbo for r_ind in eachindex(r)
			K[r_ind, vind, lmind] /= r[r_ind]
		end
	end
end
@inline function mulprefactor!(K::StructArray{<:Complex}, vindskip = 0)
	mulprefactor!(K.re, vindskip)
	mulprefactor!(K.im, vindskip)
end

function populatekernel!(::Flow, ::los_radial, K::AbstractArray{<:Complex,3},
	SHModes, Yjₒjₛ_lm_n₁n₂, Yjₒjₛ_lm_n₂n₁,
	(l, m), (jₛ, jₒ, ω),
	twoimagconjhωHγℓjₒjₛ_r₁r₂,
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
	pre⁰, pre⁺, phase)

	mode_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂), (l, m))

	Pʲₒʲₛₗₘ_₀₀_n₁n₂ = Yjₒjₛ_lm_n₁n₂[mode_ind]
	Pʲₒʲₛₗₘ_₀₀_n₂n₁ = Yjₒjₛ_lm_n₂n₁[mode_ind]

	iszero(Pʲₒʲₛₗₘ_₀₀_n₁n₂) && iszero(Pʲₒʲₛₗₘ_₀₀_n₂n₁) && return

	anyzeromom = l == 0 || jₛ == 0
	evenmom = iseven(jₒ + jₛ + l)

	zeroindT, plusindT = axes(twoimagconjhωconjHγℓjₒjₛ_r₂r₁, 2)
	minusindK, zeroindK, plusindK = axes(K, 1)
	lm_ind_K = modeindex(SHModes, (l, m))

	if evenmom
		@turbo for r_ind in eachindex(r)
			TP₁₂⁰ = pre⁰ * Pʲₒʲₛₗₘ_₀₀_n₁n₂ * twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind, zeroindT]
			TP₂₁⁰ = pre⁰ * Pʲₒʲₛₗₘ_₀₀_n₂n₁ * twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind, zeroindT]

			K[zeroindK, r_ind, lm_ind_K] += TP₂₁⁰ - TP₁₂⁰
		end
	end
	anyzeromom && return K
	@turbo for r_ind in eachindex(r)
		TP₁₂¹ = pre⁺ * Pʲₒʲₛₗₘ_₀₀_n₁n₂ * twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind, plusindT]
		TP₂₁¹ = pre⁺ * Pʲₒʲₛₗₘ_₀₀_n₂n₁ * twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind, plusindT]

		K[plusindK, r_ind, lm_ind_K] += TP₂₁¹ - TP₁₂¹
		K[minusindK, r_ind, lm_ind_K] += phase * (TP₂₁¹ - TP₁₂¹)
	end

	return K
end
function populatekernel!(::Flow, ::los_earth, K::AbstractArray{<:Complex,3},
	SHModes, Yjₒjₛ_lm_n₁n₂, Yjₒjₛ_lm_n₂n₁,
	(l, m), (jₛ, jₒ, ω),
	twoimagconjhωHγℓjₒjₛ_r₁r₂,
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
	pre⁰, pre⁺, phase)

	mode_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂), (l, m))
	lm_ind_K = modeindex(SHModes, (l, m))

	_Yjₒjₛ_lm_n₁n₂ = parent(Yjₒjₛ_lm_n₁n₂[mode_ind])
	_Yjₒjₛ_lm_n₂n₁ = parent(Yjₒjₛ_lm_n₂n₁[mode_ind])

	minusindK, zeroindK, plusindK = axes(K, 1)
	zeroindT, plusindT = axes(twoimagconjhωconjHγℓjₒjₛ_r₂r₁, 2)

	anyzeromom = l == 0 || jₛ == 0
	evenmom = iseven(jₒ + jₛ + l)
	if evenmom
		@turbo for r_ind in eachindex(r)
			TP₁₂⁰ = pre⁰ * sum(_Yjₒjₛ_lm_n₁n₂ .* twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind, zeroindT])
			TP₂₁⁰ = pre⁰ * sum(_Yjₒjₛ_lm_n₂n₁ .* twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind, zeroindT])

			K[zeroindK, r_ind, lm_ind_K] += TP₂₁⁰ - TP₁₂⁰
		end
	end
	anyzeromom && return K
	@turbo for r_ind in eachindex(r)
		TP₁₂¹ = pre⁺ * sum(_Yjₒjₛ_lm_n₁n₂ .* twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind, plusindT])
		TP₂₁¹ = pre⁺ * sum(_Yjₒjₛ_lm_n₂n₁ .* twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind, plusindT])

		K[plusindK, r_ind, lm_ind_K] += TP₂₁¹ - TP₁₂¹
		K[minusindK, r_ind, lm_ind_K] += phase * (TP₂₁¹ - TP₁₂¹)
	end

	return K
end

function populatekernelrθϕl0!(::Flow, ::los_radial, K::AbstractArray{<:Real, 3},
	SHModes, Yjₒjₛ_lm_n₁n₂, Yjₒjₛ_lm_n₂n₁,
	(l,_), (jₛ, jₒ, ω), twoimagconjhωconjHγℓjₒjₛ_r₂r₁, twoimagconjhωHγℓjₒjₛ_r₁r₂,
	pre⁰, pre⁺, phase)

	lm_ind_K = modeindex(SHModes, (l, 0))
	lm_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂), (l, 0))
	Pʲₒʲₛₗₘ_₀₀_n₁n₂ = Yjₒjₛ_lm_n₁n₂[lm_ind]
	Pʲₒʲₛₗₘ_₀₀_n₂n₁ = Yjₒjₛ_lm_n₂n₁[lm_ind]

	rindK, θindK, ϕindK = axes(K, 1)
	zeroindT, plusindT = axes(twoimagconjhωconjHγℓjₒjₛ_r₂r₁, 2)

	anyzeromom = l == 0 || jₛ == 0
	evenmom = iseven(jₒ + jₛ + l)

	if evenmom
		@turbo for r_ind in eachindex(r)
			TP₁₂⁰ = pre⁰ * Pʲₒʲₛₗₘ_₀₀_n₁n₂ * twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind, zeroindT]
			TP₂₁⁰ = pre⁰ * Pʲₒʲₛₗₘ_₀₀_n₂n₁ * twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind, zeroindT]

			K[rindK, r_ind, lm_ind_K] += real(TP₂₁⁰) - real(TP₁₂⁰)
		end
	end

	# l = 0 only has an r component
	anyzeromom && return K
	@turbo for r_ind in eachindex(r)
		TP₁₂¹ = pre⁺ * Pʲₒʲₛₗₘ_₀₀_n₁n₂ * twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind, plusindT]
		TP₂₁¹ = pre⁺ * Pʲₒʲₛₗₘ_₀₀_n₂n₁ * twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind, plusindT]

		tempre = real(TP₂₁¹) - real(TP₁₂¹)
		tempim = imag(TP₂₁¹) - imag(TP₁₂¹)

		K[θindK, r_ind, lm_ind_K] += evenmom * 2tempre
		K[ϕindK, r_ind, lm_ind_K] += !evenmom * (-2tempim)
	end

	return K
end
function populatekernelrθϕl0!(::Flow, ::los_earth, K::AbstractArray{<:Real, 3},
	SHModes, Yjₒjₛ_lm_n₁n₂, Yjₒjₛ_lm_n₂n₁,
	(l,_), (jₛ, jₒ, ω), twoimagconjhωconjHγℓjₒjₛ_r₂r₁, twoimagconjhωHγℓjₒjₛ_r₁r₂,
	pre⁰, pre⁺, phase)

	lm_ind_K = modeindex(SHModes, (l, 0))
	lm_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂), (l, 0))

	_Yjₒjₛ_lm_n₁n₂ = parent(Yjₒjₛ_lm_n₁n₂[lm_ind])
	_Yjₒjₛ_lm_n₂n₁ = parent(Yjₒjₛ_lm_n₂n₁[lm_ind])

	rindK, θindK, ϕindK = axes(K, 1)
	zeroindT, plusindT = axes(twoimagconjhωconjHγℓjₒjₛ_r₂r₁, 2)

	anyzeromom = l == 0 || jₛ == 0
	evenmom = iseven(jₒ + jₛ + l)

	if evenmom
		@turbo for r_ind in eachindex(r)
			TP₁₂⁰ = pre⁰ * sum(_Yjₒjₛ_lm_n₁n₂ .* twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind, zeroindT])
			TP₂₁⁰ = pre⁰ * sum(_Yjₒjₛ_lm_n₂n₁ .* twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind, zeroindT])

			K[rindK, r_ind, lm_ind_K] += real(TP₂₁⁰) - real(TP₁₂⁰)
		end
	end
	anyzeromom && return K
	@turbo for r_ind in eachindex(r)
		TP₁₂¹ = pre⁺ * sum(_Yjₒjₛ_lm_n₁n₂ .* twoimagconjhωHγℓjₒjₛ_r₁r₂[r_ind, plusindT])
		TP₂₁¹ = pre⁺ * sum(_Yjₒjₛ_lm_n₂n₁ .* twoimagconjhωconjHγℓjₒjₛ_r₂r₁[r_ind, plusindT])

		tempre = real(TP₂₁¹) - real(TP₁₂¹)
		tempim = imag(TP₂₁¹) - imag(TP₁₂¹)

		K[θindK, r_ind, lm_ind_K] += evenmom * 2tempre
		K[ϕindK, r_ind, lm_ind_K] += !evenmom * (-2tempim)
	end

	return K
end

reinterpretSMatrix(A::AbstractArray{<:Any, 4}) = dropdims(reinterpret(SMatrix{2,2,eltype(A),4}, reshape(A, 4, size(A)[3:4]...)), dims = 1)
reinterpretSMatrix(A::AbstractArray{<:Any, 2}) = A

function kernel_uₛₜ_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::SphericalPoint, xobs2::SphericalPoint, los::los_direction, SHModes, hω_arr,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default)

	@unpack p_Gsrc, p_Gobs1, p_Gobs2, Gfn_path_src, NGfn_files_src, Gfn_path_obs1, NGfn_files_obs1,
	Gfn_path_obs2, NGfn_files_obs2 = unpackGfnparams(p_Gsrc, r_src, p_Gobs1, radius(xobs1), p_Gobs2, radius(xobs2));

	@unpack ℓ_arr, ω_arr, Nν_Gfn, dω = p_Gsrc

	s_range = l_range(SHModes)
	s_max = maximum(s_range)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))
	obs_at_same_height = r₁_ind == r₂_ind

	ℓ_range_proc = _extremaelementrange(ℓ_ωind_iter_on_proc, dims = 1)

	# Get a list of all modes that will be accessed.
	# This can be used to open the fits files before the loops begin.
	# This will cut down on FITS IO costs
	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, NGfn_files_src)

	# Gℓ′ω(r, robs) files
	Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_obs1,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, s_max, NGfn_files_obs1)
	# Gℓ′ω(r, robs) files
	Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_obs2,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, s_max, NGfn_files_obs2)

	Kre = zeros(3, nr, length(SHModes)); # first axis is for vector indices (-1,0,1 in the helicity basis)
	K = StructArray{ComplexF64}((Kre, zero(Kre))); # Kγₗₘ(r, x₁, x₂)

	ind⁰ = 2 # index of the vector component 0

	arrs = allocatearrays(Flow(), los, obs_at_same_height);
	@unpack Gsrc, drGsrc, Gparts_r₁, Gparts_r₂,
	Gγℓjₒjₛ_r₁, Gγℓjₒjₛ_r₂, Hγℓjₒjₛ_r₁r₂, Hγℓjₒjₛ_r₂r₁ = arrs;
	@unpack Gparts_r₁_2, Gparts_r₂_2 = arrs;
	@unpack twoimagconjhωconjHγℓjₒjₛ_r₂r₁, twoimagconjhωHγℓjₒjₛ_r₁r₂ = arrs;
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁P, twoimagconjhωHγℓjₒjₛ_r₁r₂P = map(parent, (twoimagconjhωconjHγℓjₒjₛ_r₂r₁, twoimagconjhωHγℓjₒjₛ_r₁r₂));

	HT = HybridArray{Tuple{sizeindG(los)...,sizeindG(los)...,StaticArrays.Dynamic(),2}}
	H₂₁ = _structarrayparent(Hγℓjₒjₛ_r₂r₁, HT)
	H₁₂ = _structarrayparent(Hγℓjₒjₛ_r₁r₂, HT)

	twoimagconjhωconjHγℓjₒjₛ_r₂r₁S = reinterpretSMatrix(twoimagconjhωconjHγℓjₒjₛ_r₂r₁P);
	twoimagconjhωHγℓjₒjₛ_r₁r₂S = reinterpretSMatrix(twoimagconjhωHγℓjₒjₛ_r₁r₂P);

	jₒjₛ_allmodes = L2L1Triangle(ℓ_range_proc, s_max, ℓ_arr)
	jₒrange = l2_range(jₒjₛ_allmodes)
	@unpack Gobs1_cache, Gobs2_cache = arrs
	for jₒ in jₒrange
		Gobs1_cache[jₒ] = zeros(ComplexF64, axes(Gsrc))
		if !obs_at_same_height
			Gobs2_cache[jₒ] = zeros(ComplexF64, axes(Gsrc))
		end
	end

	@timeit localtimer "biposh" begin
		Y12, Y21 = los_projected_spheroidal_biposh_flippoints(xobs1, xobs2, los, SHModes, jₒjₛ_allmodes)
	end

	C = zeros(0:1, 0:s_max, l2_range(jₒjₛ_allmodes), ℓ_range_proc)

	for jₛ in axes(C, 4), jₒ in axes(C, 3), ℓ in axes(C, 2)

		C[0, ℓ, jₒ, jₛ] = if iseven(jₒ+jₛ+ℓ)
							clebschgordan(Float64, jₒ, 0, jₛ, 0, ℓ, 0)
						else
							zero(Float64)
						end

		C[1, ℓ, jₒ, jₛ] = if ℓ > 0 && jₛ > 0
							clebschgordan(Float64, jₒ, 0, jₛ, 1, ℓ, 1)
						else
							zero(Float64)
						end
	end

	# for some reason the types of these variables were not being inferred
	C⁰::Float64, C⁺::Float64 = 0, 0
	phase::Int = 1
	pre⁰::Float64, pre⁺::Float64 = 0, 0

	ω_ind_prev = -1 # something unrealistic to start off with

	# Loop over the Greenfn files
	for (jₛ, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]
		h_ω = hω_arr[ω_ind]
		dωω³Pω = dω/2π * ω^3 * Powspec(ω)
		Ωjₛ0 = Ω(jₛ, 0)

		@timeit localtimer "FITS" begin
		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (jₛ, ω_ind), NGfn_files_src, 1:2, srcindFITS(los), :, 1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (jₛ, ω_ind), NGfn_files_src, 1:2, srcindFITS(los), :, 2)
		end # timer

		Gα₂r_r₁_rₛ = αrcomp(Gsrc, r₁_ind, los)
		Gα₂r_r₂_rₛ = αrcomp(Gsrc, r₂_ind, los)

		for jₒ in l2_range(jₒjₛ_allmodes, jₛ)

			jₒjₛ_ind = modeindex(jₒjₛ_allmodes, (jₒ, jₛ))

			Yjₒjₛ_lm_n₁n₂ = Y12[jₒjₛ_ind]
			Yjₒjₛ_lm_n₂n₁ = Y21[jₒjₛ_ind]

			(Yjₒjₛ_lm_n₁n₂ === nothing || Yjₒjₛ_lm_n₂n₁ === nothing) && continue

			# Change shmodes to ML to avoid recomputing the radial term for same l
			modesML = intersect(ML(firstshmodes(Yjₒjₛ_lm_n₁n₂)), ML(SHModes))
			modesML === nothing && continue

			Gobs1 = Gobs1_cache[jₒ]
			@timeit localtimer "FITS" begin
			# Green functions based at the observation point
			if ω_ind != ω_ind_prev || jₒ ∉ l2_range(jₒjₛ_allmodes, jₛ-1)
				read_Gfn_file_at_index!(Gobs1, Gfn_fits_files_obs1,
				(ℓ_arr, 1:Nν_Gfn), (jₒ, ω_ind), NGfn_files_obs1, 1:2, srcindFITS(los), :, 1)
			end
			end # timer

			@timeit localtimer "radial term 1" begin
			Gⱼₒⱼₛω_u⁰⁺_firstborn!(Gparts_r₁, Gparts_r₁_2, Gsrc, drGsrc, jₛ, Gobs1, jₒ, los)
			end # timer

			if !obs_at_same_height
				Gobs2 = Gobs2_cache[jₒ]
				@timeit localtimer "FITS" begin
				if ω_ind != ω_ind_prev || jₒ ∉ l2_range(jₒjₛ_allmodes, jₛ-1)
					read_Gfn_file_at_index!(Gobs2, Gfn_fits_files_obs2,
					(ℓ_arr, 1:Nν_Gfn), (jₒ, ω_ind), NGfn_files_obs2, 1:2, srcindFITS(los), :, 1)
				end
				end # timer

				@timeit localtimer "radial term 1" begin
				Gⱼₒⱼₛω_u⁰⁺_firstborn!(Gparts_r₂, Gparts_r₂_2, Gsrc, drGsrc, jₛ, Gobs2, jₒ, los)
				end #timer
			end

			l_prev = -1 # something unrealistic to start off

			for (l, m) in modesML
				# Check if triangle condition is satisfied
				# The outer loop is over all possible jₛ and jₒ for a given l_max
				# Not all combinations would contribute towards a specific l
				δ(jₛ, jₒ, l) || continue

				if l != l_prev
					@timeit localtimer "radial term 2" begin
						@timeit localtimer "H" begin
							Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(Hγℓjₒjₛ_r₁r₂, Gparts_r₁_2, Gγℓjₒjₛ_r₁, Gα₂r_r₂_rₛ, jₒ, jₛ, l)

							if !obs_at_same_height
								Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(Hγℓjₒjₛ_r₂r₁, Gparts_r₂_2, Gγℓjₒjₛ_r₂, Gα₂r_r₁_rₛ, jₒ, jₛ, l)
							end
						end

						@timeit localtimer "T" begin
							@turbo for I in CartesianIndices(H₂₁)
								twoimagconjhωconjHγℓjₒjₛ_r₂r₁P[I] = -2imag(h_ω * H₂₁[I])
								twoimagconjhωHγℓjₒjₛ_r₁r₂P[I] = 2imag(conj(h_ω) * H₁₂[I])
							end
						end

						C⁰ = C[0, l, jₒ, jₛ]
						C⁺ = C[1, l, jₒ, jₛ]
						phase = (-1)^(jₒ + jₛ + l)
						coeffj = √((2jₒ+1)*(2jₛ+1)/π/(2l+1))

						pre⁰ = dωω³Pω * coeffj * C⁰
						pre⁺ = dωω³Pω * coeffj * Ωjₛ0 * C⁺
						l_prev = l
					end
				end

				@timeit localtimer "kernel" begin
				populatekernel!(Flow(), los, K,
					SHModes,
					Yjₒjₛ_lm_n₁n₂, Yjₒjₛ_lm_n₂n₁,
					(l, m), (jₛ, jₒ, ω),
					twoimagconjhωHγℓjₒjₛ_r₁r₂S,
					twoimagconjhωconjHγℓjₒjₛ_r₂r₁S,
					pre⁰, pre⁺, phase)
				end #timer
			end
		end

		ω_ind_prev = ω_ind
	end

	map(closeGfnfits, (Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2))

	_K = permutedims(K, [2,1,3])
	mulprefactor!(_K, ind⁰)
	AK = Array(no_offset_view(_K))

	return AK
end

# Compute Kₛₜ first and then compute Kₛ₀_rθϕ from that
function kernel_uₛ₀_rθϕ_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::SphericalPoint, xobs2::SphericalPoint, los::los_direction, SHModes, hω_arr,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default)

	K = kernel_uₛₜ_partial(localtimer, ℓ_ωind_iter_on_proc,
		xobs1, xobs2, los, SHModes, hω_arr,
		p_Gsrc, p_Gobs1, p_Gobs2, r_src)

	K_rθϕ = zeros(size(K, 1), 3, size(K, 3))

	ind⁰, ind⁺ = UnitRange(axes(K,2))[2:3]

	@turbo for st_ind in UnitRange(axes(K, 3))
		for r_ind in UnitRange(axes(K, 1))
			# r-component
			K_rθϕ[r_ind, 1, st_ind] = real(K[r_ind, ind⁰, st_ind])
		end
		for r_ind in UnitRange(axes(K, 1))
			# θ-component
			K_rθϕ[r_ind, 2, st_ind] = 2real(K[r_ind, ind⁺, st_ind])
		end
		for r_ind in UnitRange(axes(K, 1))
			# ϕ-component
			K_rθϕ[r_ind, 3, st_ind] = -2imag(K[r_ind, ind⁺, st_ind])
		end
	end

	K_rθϕ
end

# Compute Kₛ₀_rθϕ directly
function kernel_uₛ₀_rθϕ_partial_2(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::SphericalPoint, xobs2::SphericalPoint, los::los_direction, SHModes, hω_arr,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default)

	@unpack p_Gsrc, p_Gobs1, p_Gobs2, Gfn_path_src, NGfn_files_src, Gfn_path_obs1, NGfn_files_obs1,
	Gfn_path_obs2, NGfn_files_obs2 = unpackGfnparams(p_Gsrc, r_src, p_Gobs1, radius(xobs1), p_Gobs2, radius(xobs2))

	@unpack ℓ_arr, ω_arr, Nν_Gfn, dω = p_Gsrc

	s_max = l_max(SHModes)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))
	obs_at_same_height = r₁_ind == r₂_ind

	ℓ_range_proc = _extremaelementrange(ℓ_ωind_iter_on_proc, dims = 1)

	# Get a list of all modes that will be accessed.
	# This can be used to open the fits files before the loops begin.
	# This will cut down on FITS IO costs
	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, NGfn_files_src)

	# Gℓ′ω(r, robs) files
	Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_obs1,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, s_max, NGfn_files_obs1)
	# Gℓ′ω(r, robs) files
	Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_obs2,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, s_max, NGfn_files_obs2)

	K = zeros(3, nr, length(SHModes)) # Kγₗₘ(r, x₁, x₂)
	indr = 1

	arrs = allocatearrays(Flow(), los, obs_at_same_height)
	@unpack Gsrc, drGsrc, Gobs1, Gobs2, Gparts_r₁, Gparts_r₂, Gparts_r₁_2, Gparts_r₂_2,
	Gγℓjₒjₛ_r₁, Gγℓjₒjₛ_r₂, Hγℓjₒjₛ_r₁r₂, Hγℓjₒjₛ_r₂r₁ = arrs
	@unpack twoimagconjhωconjHγℓjₒjₛ_r₂r₁, twoimagconjhωHγℓjₒjₛ_r₁r₂ = arrs
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁P, twoimagconjhωHγℓjₒjₛ_r₁r₂P = map(parent, (twoimagconjhωconjHγℓjₒjₛ_r₂r₁, twoimagconjhωHγℓjₒjₛ_r₁r₂))

	HT = HybridArray{Tuple{sizeindG(los)...,sizeindG(los)...,StaticArrays.Dynamic(),2}}
	H₂₁R = HT(parent(Hγℓjₒjₛ_r₂r₁.re))
	H₂₁I = HT(parent(Hγℓjₒjₛ_r₂r₁.im))
	H₁₂R = HT(parent(Hγℓjₒjₛ_r₁r₂.re))
	H₁₂I = HT(parent(Hγℓjₒjₛ_r₁r₂.im))

	twoimagconjhωconjHγℓjₒjₛ_r₂r₁S = reinterpretSMatrix(twoimagconjhωconjHγℓjₒjₛ_r₂r₁P)
	twoimagconjhωHγℓjₒjₛ_r₁r₂S = reinterpretSMatrix(twoimagconjhωHγℓjₒjₛ_r₁r₂P)

	jₒjₛ_allmodes = L2L1Triangle(ℓ_range_proc, s_max, ℓ_arr)

	@timeit localtimer "biposh" begin
		Y12, Y21 = los_projected_spheroidal_biposh_flippoints(xobs1, xobs2, los, SHModes, jₒjₛ_allmodes)
	end

	C = zeros(0:1, 0:s_max, l2_range(jₒjₛ_allmodes), ℓ_range_proc)

	@timeit localtimer "CG" begin
	for jₛ in axes(C, 4), jₒ in axes(C, 3), ℓ in axes(C, 2)

		C[0, ℓ, jₒ, jₛ] = if iseven(jₒ+jₛ+ℓ)
							clebschgordan(Float64, jₒ, 0, jₛ, 0, ℓ, 0)
						else
							zero(Float64)
						end

		C[1, ℓ, jₒ, jₛ] = if ℓ > 0 && jₛ > 0
							clebschgordan(Float64, jₒ, 0, jₛ, 1, ℓ, 1)
						else
							zero(Float64)
						end
	end
	end # timer

	C⁰, C⁺ = zero(Float64), zero(Float64)
	phase = 1
	pre⁰, pre⁺ = zero(Float64), zero(Float64)

	# Loop over the Greenfn files
	for (jₛ, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]
		h_ω = hω_arr[ω_ind]
		dωω³Pω = dω/2π * ω^3 * Powspec(ω)
		Ωjₛ0 = Ω(jₛ, 0)

		@timeit localtimer "FITS" begin
		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (jₛ, ω_ind), NGfn_files_src, 1:2, srcindFITS(los), :, 1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (jₛ, ω_ind), NGfn_files_src, 1:2, srcindFITS(los), :, 2)
		end # timer

		Gα₂r_r₁_rₛ = αrcomp(Gsrc, r₁_ind, los)
		Gα₂r_r₂_rₛ = αrcomp(Gsrc, r₂_ind, los)

		for jₒ in l2_range(jₒjₛ_allmodes, jₛ)

			Yjₒjₛ_lm_n₁n₂ = Y12[(jₒ, jₛ)]
			Yjₒjₛ_lm_n₂n₁ = Y21[(jₒ, jₛ)]

			(Yjₒjₛ_lm_n₁n₂ === nothing || Yjₒjₛ_lm_n₂n₁ === nothing) && continue

			@timeit localtimer "FITS" begin
			# Green functions based at the observation point
			read_Gfn_file_at_index!(Gobs1, Gfn_fits_files_obs1,
			(ℓ_arr, 1:Nν_Gfn), (jₒ, ω_ind), NGfn_files_obs1, 1:2, srcindFITS(los), :, 1)
			end # timer

			@timeit localtimer "radial term 1" begin
			Gⱼₒⱼₛω_u⁰⁺_firstborn!(Gparts_r₁, Gparts_r₁_2, Gsrc, drGsrc, jₛ, Gobs1, jₒ, los)
			end # timer

			if !obs_at_same_height
				@timeit localtimer "FITS" begin
					read_Gfn_file_at_index!(Gobs2, Gfn_fits_files_obs2,
					(ℓ_arr, 1:Nν_Gfn), (jₒ, ω_ind), NGfn_files_obs2, 1:2, srcindFITS(los), :, 1)
				end # timer

				@timeit localtimer "radial term 1" begin
				Gⱼₒⱼₛω_u⁰⁺_firstborn!(Gparts_r₂, Gparts_r₂_2, Gsrc, drGsrc, jₛ, Gobs2, jₒ, los)
				end #timer
			end

			for (l, m) in firstshmodes(Yjₒjₛ_lm_n₁n₂)
				# Check if triangle condition is satisfied
				# The outer loop is over all possible jₛ and jₒ for a given s_max
				# Not all combinations would contribute towards a specific l
				δ(jₛ, jₒ, l) || continue

				@timeit localtimer "radial term 2" begin
					@timeit localtimer "H" begin
						Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(Hγℓjₒjₛ_r₁r₂, Gparts_r₁_2, Gγℓjₒjₛ_r₁, Gα₂r_r₂_rₛ, jₒ, jₛ, l)

						if !obs_at_same_height
							Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(Hγℓjₒjₛ_r₂r₁, Gparts_r₂_2, Gγℓjₒjₛ_r₂, Gα₂r_r₁_rₛ, jₒ, jₛ, l)
						end
					end
					# @. twoimagconjhωconjHγℓjₒjₛ_r₂r₁ = -2imag(hω * Hγℓjₒjₛ_r₂r₁)
					# @. twoimagconjhωHγℓjₒjₛ_r₁r₂ = 2imag(conjhω * Hγℓjₒjₛ_r₁r₂)
					@timeit localtimer "T" begin
						@turbo for I in CartesianIndices(H₂₁R)
							twoimagconjhωconjHγℓjₒjₛ_r₂r₁P[I] = -2(real(h_ω) * H₂₁I[I] + imag(h_ω) * H₂₁R[I])
							twoimagconjhωHγℓjₒjₛ_r₁r₂P[I] = 2(real(h_ω) * H₁₂I[I] - imag(h_ω) * H₁₂R[I])
						end
					end
				end #timer

				C⁰ = C[0, l, jₒ, jₛ]
				C⁺ = C[1, l, jₒ, jₛ]
				phase = (-1)^(jₒ + jₛ + l)
				coeffj = √((2jₒ+1)*(2jₛ+1)/π/(2l+1))

				pre⁰ = dωω³Pω * coeffj * C⁰
				pre⁺ = dωω³Pω * coeffj * Ωjₛ0 * C⁺

				@timeit localtimer "kernel" begin
				populatekernelrθϕl0!(Flow(), los, K, SHModes, Yjₒjₛ_lm_n₁n₂, Yjₒjₛ_lm_n₂n₁,
				(l, m), (jₛ, jₒ, ω), twoimagconjhωconjHγℓjₒjₛ_r₂r₁S, twoimagconjhωHγℓjₒjₛ_r₁r₂S,
				pre⁰, pre⁺, phase)
				end #timer
			end
		end
	end

	map(closeGfnfits, (Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2))
	_K = Base.PermutedDimsArray(K, (2,1,3))
	mulprefactor!(_K, indr)

	return Array(_K)
end

function generatefitsheader(xobs1, xobs2, SHModes, j_range, ν_ind_range)
	FITSHeader(["r1","th1","phi1","r2","th2","phi2",
		"l_min","m_min","l_max","m_max",
		"j_min","j_max","nui_min","nui_max"],
		Any[float(radius(xobs1)), float(xobs1.θ), float(xobs1.ϕ),
		float(radius(xobs2)), float(xobs2.θ), float(xobs2.ϕ),
		l_min(SHModes), m_min(SHModes), l_max(SHModes), m_max(SHModes),
		minimum(j_range), maximum(j_range),
		minimum(ν_ind_range), maximum(ν_ind_range)],
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

function modetag(j_range, SHModes)
   "jmax$(maximum(j_range))_lmax$(l_max(SHModes))_mmax$(m_max(SHModes))"
end
function kernelfilenameuₛₜ(m::SeismicMeasurement, los::los_direction, j_range, SHModes, tag="")
	mtag = m isa TravelTimes ? "δτ" : "A"
	lostag = los isa los_radial ? "" : "_los"
	"Kst_$(mtag)_u_$(modetag(j_range, SHModes))$(lostag)"*string(tag)*".fits"
end

function kernel_uₛₜ(comm, m::SeismicMeasurement, xobs1, xobs2,
	los::los_direction = los_radial(); kwargs...)

	SHModes = getkernelmodes(; kwargs...)

	r_src, = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc, p_Gobs1, p_Gobs2 = read_parameters_for_points(xobs1, xobs2; kwargs...)
	@unpack ν_arr, ℓ_arr = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr,ν_arr; kwargs...)
	j_range, ν_ind_range = iters

	hω_arr = get(kwargs, :hω) do
		h = hω(comm, m, xobs1, xobs2, los; kwargs..., print_timings = false)
		_broadcast(h, 0, comm)
	end

	K_δτ_uₛₜ = pmapsum(comm, kernel_uₛₜ_partial, iters,
		xobs1, xobs2, los, SHModes, hω_arr, p_Gsrc, p_Gobs1, p_Gobs2, r_src)

	K_δτ_uₛₜ === nothing  && return nothing
	K = OffsetArray(K_δτ_uₛₜ, :, -1:1, :)

	if get(kwargs, :save, true)

		filepath = joinpath(SCRATCH_kerneldir[], kernelfilenameuₛₜ(m, los, j_range, SHModes))

		header = generatefitsheader(xobs1, xobs2, SHModes, j_range, ν_ind_range)

		FITS(filepath,"w") do f
			FITSIO.write(f, reinterpret_as_float(K_δτ_uₛₜ), header = header)
		end
	end

	SHArray(K, (axes(K)[1:2]..., SHModes))
end

kernelfilenameuₛ₀rθϕ(::TravelTimes, ::los_radial) = "Kl0_δτ_u_rθϕ.fits"
kernelfilenameuₛ₀rθϕ(::TravelTimes, ::los_earth) = "Kl0_δτ_u_rθϕ_los.fits"
kernelfilenameuₛ₀rθϕ(::Amplitudes, ::los_radial) = "Kl0_A_u_rθϕ.fits"
kernelfilenameuₛ₀rθϕ(::Amplitudes, ::los_earth) = "Kl0_A_u_rθϕ_los.fits"

# Compute Kₛₜ first and then compute Kₛ₀_rθϕ from that
function kernel_uₛ₀_rθϕ(comm, m::SeismicMeasurement, xobs1, xobs2, los::los_direction = los_radial(); kwargs...)
	_SHModes = getkernelmodes(; kwargs...)
	SHModes = LM(l_range(_SHModes), 0)

	r_src, = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc, p_Gobs1, p_Gobs2 = read_parameters_for_points(xobs1, xobs2; kwargs...)
	@unpack ν_arr, ℓ_arr = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)
	j_range, ν_ind_range = iters

	hω_arr = get(kwargs, :hω) do
		h = hω(comm, m, xobs1, xobs2, los; kwargs..., print_timings = false)
		_broadcast(h, 0, comm)
	end

	K_δτ_uₛ₀ = pmapsum(comm, kernel_uₛ₀_rθϕ_partial, iters,
		xobs1, xobs2, los, SHModes, hω_arr, p_Gsrc, p_Gobs1, p_Gobs2, r_src)

	K_δτ_uₛ₀ === nothing  && return nothing

	if get(kwargs, :save, true)
		filename = joinpath(SCRATCH_kerneldir[], kernelfilenameuₛ₀rθϕ(m, los))

		header = generatefitsheader(xobs1, xobs2, SHModes, j_range, ν_ind_range)

		FITS(filename,"w") do f
			FITSIO.write(f, reinterpret_as_float(K_δτ_uₛ₀), header = header)
		end
	end

	SHArray(K_δτ_uₛ₀, (axes(K_δτ_uₛ₀)[1:2]..., SHModes))
end

# Compute Kₛ₀_rθϕ directly
function kernel_uₛ₀_rθϕ_2(comm, m::SeismicMeasurement, xobs1, xobs2, los::los_direction = los_radial(); kwargs...)
	_SHModes = getkernelmodes(; kwargs...)
	SHModes = LM(l_range(_SHModes), 0)

	r_src, = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc, p_Gobs1, p_Gobs2 = read_parameters_for_points(xobs1, xobs2; kwargs...)
	@unpack ν_arr, ℓ_arr = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr,ν_arr; kwargs...)
	j_range, ν_ind_range = iters

	hω_arr = get(kwargs, :hω) do
		h = hω(comm, m, xobs1, xobs2, los; kwargs..., print_timings = false)
		_broadcast(h, 0, comm)
	end

	K_δτ_uₛ₀ = pmapsum(comm, kernel_uₛ₀_rθϕ_partial_2, iters,
		xobs1, xobs2, los, SHModes, hω_arr, p_Gsrc, p_Gobs1, p_Gobs2, r_src)

	K_δτ_uₛ₀ === nothing  && return nothing
	if get(kwargs, :save, true)
		filename = joinpath(SCRATCH_kerneldir[], kernelfilenameuₛ₀rθϕ(m, los))

		header = generatefitsheader(xobs1, xobs2, SHModes, j_range, ν_ind_range)

		FITS(filename,"w") do f
			FITSIO.write(f, reinterpret_as_float(K_δτ_uₛ₀), header = header)
		end
	end

	SHArray(K_δτ_uₛ₀, (axes(K_δτ_uₛ₀)[1:2]..., SHModes))
end

function kernel_ψϕₛ₀(comm, m::SeismicMeasurement, xobs1, xobs2, los::los_direction = los_radial(); kwargs...)
	Kv = kernel_uₛ₀_rθϕ(comm, m, xobs1, xobs2, los; kwargs..., save = false)
	Kv === nothing && return nothing
	kernel_ψϕₛ₀(Kv; kwargs...)
end

function kernel_ψϕₛ₀(Kv::AbstractArray; kwargs...)
	mode_range = firstshmodes(Kv);
	Kψϕ = zeros(axes(Kv, 1), mode_range);
	kernel_ψϕₛ₀!(Kψϕ, Kv; save = get(kwargs, :save, true))

	if get(kwargs, :save, true)
		filename = joinpath(SCRATCH_kerneldir[], "Kpsi_imag.fits")
		FITS(filename,"w") do f
			write(f, reinterpret_as_float(Kψϕ))
		end
	end

	return Kψϕ
end

function kernel_ψϕₛ₀!(Kψϕ::SHArray, Kv::SHArray; kwargs...)
	mode_range = firstshmodes(Kv)

	for l in l_range(mode_range)
		@views Kψϕ[:, (l, 0)] .= -(ddr*(Kv[:, 2, (l, 0)]./ρ) .+
			@. (Kv[:, 2, (l, 0)] - 2Ω(l, 0)*Kv[:, 1, (l, 0)])/(ρ*r) )
	end

	return Kψϕ
end

#################################################################################################################
# Validation for isotropic sound speed perturbation
#################################################################################################################

function populatekernelvalidation!(::SoundSpeed, ::los_radial, K::AbstractVector,
	(ℓ, ω), Y12ℓ, conjhω, H¹₁jjω_r₁r₂, H¹₁jj_r₂r₁, dω)

	pre = 1/√(4π) * dω/2π * ω^2 * Powspec(ω) * Y12ℓ
	@. K += pre * 2real(conjhω * ( H¹₁jjω_r₁r₂ + conj(H¹₁jj_r₂r₁) ) )
end

function populatekernelvalidation!(::SoundSpeed, ::los_radial, K::AbstractMatrix,
	(ℓ, ω), Y12ℓ, conjhω, H¹₁jjω_r₁r₂::AbstractVector{<:Complex},
	H¹₁jj_r₂r₁::AbstractVector{<:Complex}, dω)

	pre = 1/√(4π) * dω/2π * ω^2 * Powspec(ω)

	for n2ind in UnitRange(axes(K, 2))
		temp_ωℓ_n2 = pre * Y12ℓ[n2ind]
		conjhω_n2 = conjhω[n2ind]
		for r_ind in UnitRange(axes(K, 1))
			K[r_ind, n2ind] += temp_ωℓ_n2 *
				2real(conjhω_n2 * ( H¹₁jjω_r₁r₂[r_ind] + conj(H¹₁jj_r₂r₁[r_ind]) ) )
		end
	end
end

function populatekernelvalidation!(::SoundSpeed, ::los_earth, K::AbstractVector,
	(j, ω), Y12j, conjhω, Hjj_r₁r₂, Hjj_r₂r₁, dω)

	pre = dω/2π * ω^2 * Powspec(ω) * √(1/π)

	for r_ind in UnitRange(axes(K, 1))
		for α₂ in 0:1, α₁ in 0:1

			llY = pre * real(Y12j[α₁, α₂])
			iszero(llY) && continue

			K[r_ind] += real(conjhω *
				( Hjj_r₁r₂[α₁, α₂, r_ind] + conj(Hjj_r₂r₁[α₂, α₁, r_ind]) ) ) * llY
		end
	end
end

function populatekernelvalidation!(::SoundSpeed, ::los_earth, K::AbstractMatrix,
	(j, ω), Y12j, conjhω, Hjjω_r₁r₂, Hjj_r₂r₁, dω)

	pre = dω/2π * ω^2 * Powspec(ω) * √(1/π)

	for n2ind in axes(K, 2)
		Yjn₂ = Y12j[n2ind]
		conjhω_n2 = conjhω[n2ind]

		for r_ind in UnitRange(axes(K, 1))
			for α₂ in 0:1, α₁ in 0:1

				llY = pre * real(Yjn₂[α₁, α₂])
				iszero(llY) && continue

				K[r_ind, n2ind] += real(conjhω_n2 *
					( Hjjω_r₁r₂[α₁, α₂, r_ind] + conj(Hjj_r₂r₁[α₂, α₁, r_ind]) ) ) * llY
			end
		end
	end
end

function kernel_δc₀₀_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::SphericalPoint, xobs2::SphericalPoint, los::los_direction, hω,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default)

	@unpack p_Gsrc, p_Gobs1, p_Gobs2, Gfn_path_src, NGfn_files_src,
	Gfn_path_obs1, NGfn_files_obs1, Gfn_path_obs2, NGfn_files_obs2 =
		unpackGfnparams(p_Gsrc, r_src, p_Gobs1, radius(xobs1), p_Gobs2, radius(xobs2))

	@unpack ℓ_arr, ω_arr, Nν_Gfn, dω = p_Gsrc

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2 =
		Gfn_fits_files.((Gfn_path_src, Gfn_path_obs1, Gfn_path_obs2),
			((ℓ_arr, 1:Nν_Gfn),), (ℓ_ωind_iter_on_proc,),
			(NGfn_files_src, NGfn_files_obs1, NGfn_files_obs2))

	K = zeros(nr)

	arrs = allocatearrays(SoundSpeed(), los, r₁_ind == r₂_ind);
	@unpack Gsrc, drGsrc, divGsrc, Gobs1, drGobs1, Gobs2, divGobs = arrs;
	fjj_r₁_rsrc, fjj_r₂_rsrc = arrs.fjₒjₛ_r₁_rsrc, arrs.fjₒjₛ_r₂_rsrc;
	H¹₁jj_r₁r₂, H¹₁jj_r₂r₁ = arrs.Hjₒjₛω_r₁r₂, arrs.Hjₒjₛω_r₂r₁;

	Y12 = los_projected_biposh_spheroidal(computeY₀₀, xobs1, xobs2, los, ℓ_ωind_iter_on_proc)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]
		conjhω = conj(hω[ω_ind])

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, 1:2, srcindFITS(los), :, 1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, 1:1, srcindFITS(los), :, 2)

		Gγr_r₁_rsrc = αrcomp(Gsrc, r₁_ind, los)
		Gγr_r₂_rsrc = αrcomp(Gsrc, r₂_ind, los)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1, Gfn_fits_files_obs1,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs1, 1:2, srcindFITS(los), :, 1)

		# Derivative of Green function about receiver location
		read_Gfn_file_at_index!(drGobs1, Gfn_fits_files_obs1,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs1, 1:1, srcindFITS(los), :, 2)

		radial_fn_isotropic_δc_firstborn!(fjj_r₁_rsrc,
			Gsrc, drGsrc, divGsrc, Gobs1, drGobs1, divGobs, ℓ)

		Hjₒjₛω!(H¹₁jj_r₁r₂, fjj_r₁_rsrc, Gγr_r₂_rsrc)

		if r₁_ind != r₂_ind
			# Green function about receiver location
			read_Gfn_file_at_index!(Gobs2, Gfn_fits_files_obs2,
				(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs2, 1:2, srcindFITS(los), :, 1)

			# Derivative of Green function about receiver location
			read_Gfn_file_at_index!(drGobs2, Gfn_fits_files_obs2,
				(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs2, 1:1, srcindFITS(los), :, 2)

			radial_fn_isotropic_δc_firstborn!(fjj_r₂_rsrc,
				Gsrc, drGsrc, divGsrc, Gobs2, drGobs2, divGobs, ℓ)

			Hjₒjₛω!(H¹₁jj_r₂r₁, fjj_r₂_rsrc, Gγr_r₁_rsrc)
		end

		populatekernelvalidation!(SoundSpeed(), los, K,
		(ℓ, ω), Y12[ℓ], conjhω, H¹₁jj_r₁r₂, H¹₁jj_r₂r₁, dω)
	end

	map(closeGfnfits, (Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2))
	return K
end

function kernel_δc₀₀_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	nobs1::Point2D, nobs2_arr::Vector{<:Point2D}, los::los_direction, hω_arr,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = p_Gobs1,
	r_src = r_src_default)

	hω_arr = permutedims(hω_arr) #  convert to (n2, ω)

	r_obs_ind = radial_grid_index(nobs1)

	@unpack p_Gsrc, p_Gobs1, Gfn_path_src, NGfn_files_src =
		unpackGfnparams(p_Gsrc, r_src, p_Gobs1, r_obs_default, p_Gobs2, r_obs_default)

	Gfn_path_obs, NGfn_files_obs = p_Gobs1.path, p_Gobs1.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn, dω = p_Gsrc

	Gfn_fits_files_src, Gfn_fits_files_obs =
		Gfn_fits_files.((Gfn_path_src, Gfn_path_obs),
			((ℓ_arr, 1:Nν_Gfn),), (ℓ_ωind_iter_on_proc,),
			(NGfn_files_src, NGfn_files_obs))

	K = zeros(nr, length(nobs2_arr))

	arrs = allocatearrays(SoundSpeed(), los, true)
	@unpack Gsrc, drGsrc, divGsrc, divGobs = arrs
	Gobs, drGobs = arrs.Gobs1, arrs.drGobs1
	f_robs_rsrc, Hjj_robs_rsrc = arrs.fjₒjₛ_r₁_rsrc, arrs.Hjₒjₛω_r₁r₂

	Y12 = los_projected_biposh_spheroidal(computeY₀₀, nobs1, nobs2_arr, los, ℓ_ωind_iter_on_proc)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]
		conjhω = conj(@view hω_arr[:, ω_ind])

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, 1:2, srcindFITS(los), :, 1)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, 1:1, srcindFITS(los), :, 2)

		Gγr_robs_rsrc = αrcomp(Gsrc, r_obs_ind, los)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs, Gfn_fits_files_obs,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs, 1:2, srcindFITS(los), :, 1)

		# Derivative of Green function about receiver location
		read_Gfn_file_at_index!(drGobs, Gfn_fits_files_obs,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs, 1:1, srcindFITS(los), :, 2)

		radial_fn_isotropic_δc_firstborn!(f_robs_rsrc,
			Gsrc, drGsrc, divGsrc, Gobs, drGobs, divGobs, ℓ)

		Hjₒjₛω!(Hjj_robs_rsrc, f_robs_rsrc, Gγr_robs_rsrc)

		populatekernelvalidation!(SoundSpeed(), los, K,
			(ℓ, ω), (@view Y12[:, ℓ]), conjhω, Hjj_robs_rsrc, Hjj_robs_rsrc, dω)
	end

	map(closeGfnfits, (Gfn_fits_files_src, Gfn_fits_files_obs))
	return K
end

kernelfilenameδc₀₀(::TravelTimes, ::los_radial) = "K_δτ_δc₀₀.fits"
kernelfilenameδc₀₀(::TravelTimes, ::los_earth) = "K_δτ_δc₀₀_los.fits"
kernelfilenameδc₀₀(::Amplitudes, ::los_radial) = "K_A_δc₀₀.fits"
kernelfilenameδc₀₀(::Amplitudes, ::los_earth) = "K_A_δc₀₀_los.fits"

function kernel_δc₀₀(comm, m::SeismicMeasurement, xobs1, xobs2, los::los_direction = los_radial(); kwargs...)
	r_src, = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc, p_Gobs1, p_Gobs2 = read_parameters_for_points(xobs1, xobs2; kwargs..., c_scale = 1)
	@unpack ν_arr, ℓ_arr = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr,ν_arr; kwargs...)

	hω_arr = get(kwargs, :hω) do
		h = hω(comm, m, xobs1, xobs2, los; kwargs..., print_timings = false)
		_broadcast(h, 0, comm)
	end

	K = pmapsum(comm, kernel_δc₀₀_partial, iters,
		xobs1, xobs2, los, hω_arr, p_Gsrc, p_Gobs1, p_Gobs2, r_src)
	K === nothing && return nothing

	if get(kwargs, :save, true)
		filename = joinpath(SCRATCH_kerneldir[], kernelfilenameδc₀₀(m, los))
		FITS(filename,"w") do f
			write(f, reinterpret_as_float(K))
		end
	end

	return K
end

########################################################################################
# Sound-speed kernels
########################################################################################

function populatekernel!(::SoundSpeed, ::los_radial, K::StructArray{<:Complex}, SHModes, Yjₒjₛ_n1n2, Yjₒjₛ_n2n1,
	(l, m), (jₛ, jₒ, ω), pre, tworealconjhωHjₒjₛω_r₁r₂, tworealconjhωconjHjₒjₛω_r₂r₁)

	lm_ind = modeindex(firstshmodes(Yjₒjₛ_n1n2), (l, m))
	lm_ind_K = modeindex(SHModes, (l, m))

	_Yjₒjₛ_lm_n₁n₂ = Yjₒjₛ_n1n2[lm_ind]
	_Yjₒjₛ_lm_n₂n₁ = Yjₒjₛ_n2n1[lm_ind]

	@turbo for r_ind in eachindex(r)
		T₁₂ = pre * conj(_Yjₒjₛ_lm_n₁n₂) * tworealconjhωHjₒjₛω_r₁r₂[r_ind]
		T₂₁ = pre * conj(_Yjₒjₛ_lm_n₂n₁) * tworealconjhωconjHjₒjₛω_r₂r₁[r_ind]

		K[r_ind, lm_ind_K] += T₂₁ + T₁₂
	end

	return K
end

function populatekernel!(::SoundSpeed, ::los_earth, K::StructArray{<:Complex}, SHModes, Yjₒjₛ_lm_n₁n₂, Yjₒjₛ_lm_n₂n₁,
	(l, m), (jₛ, jₒ, ω), pre, tworealconjhωHjₒjₛω_r₁r₂, tworealconjhωconjHjₒjₛω_r₂r₁)

	lm_ind = modeindex(firstshmodes(Yjₒjₛ_lm_n₁n₂), (l, m))
	lm_ind_K = modeindex(SHModes, (l, m))

	_Yjₒjₛ_lm_n₁n₂ = parent(Yjₒjₛ_lm_n₁n₂[lm_ind])
	_Yjₒjₛ_lm_n₂n₁ = parent(Yjₒjₛ_lm_n₂n₁[lm_ind])

	@turbo for r_ind in eachindex(r)
		T₁₂ = pre * sum(conj(_Yjₒjₛ_lm_n₁n₂) .* tworealconjhωHjₒjₛω_r₁r₂[r_ind])
		T₂₁ = pre * sum(conj(_Yjₒjₛ_lm_n₂n₁) .* tworealconjhωconjHjₒjₛω_r₂r₁[r_ind])

		K[r_ind, lm_ind_K] += T₂₁ + T₁₂
	end
	return K
end

reinterpretSMatrix(A::AbstractArray{<:Any, 3}) = dropdims(reinterpret(SMatrix{2,2,eltype(A),4}, reshape(A, 4, size(A, 3))), dims = 1)
reinterpretSMatrix(A::AbstractVector) = A

function kernel_δcₛₜ_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::SphericalPoint, xobs2::SphericalPoint, los::los_direction, SHModes, hω_arr,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default)

	@unpack p_Gsrc, p_Gobs1, p_Gobs2, Gfn_path_src, NGfn_files_src,
	Gfn_path_obs1, NGfn_files_obs1, Gfn_path_obs2, NGfn_files_obs2 =
		unpackGfnparams(p_Gsrc, r_src, p_Gobs1, radius(xobs1), p_Gobs2, radius(xobs2))

	@unpack ℓ_arr, ω_arr, Nν_Gfn, dω = p_Gsrc

	s_max = l_max(SHModes)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))
	obs_at_same_height = r₁_ind == r₂_ind

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, NGfn_files_src)

	# Gℓ′ω(r, robs) files
	Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_obs1,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, s_max, NGfn_files_obs1)

	Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_obs2,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, s_max, NGfn_files_obs2)

	arrs = allocatearrays(SoundSpeed(), los, obs_at_same_height)
	@unpack Gsrc, drGsrc, Gobs1, drGobs1, Gobs2, drGobs2, divGsrc, divGobs = arrs;
	@unpack fjₒjₛ_r₁_rsrc, fjₒjₛ_r₂_rsrc, Hjₒjₛω_r₁r₂, Hjₒjₛω_r₂r₁ = arrs;
	@unpack Gobs1_cache, drGobs1_cache, Gobs2_cache, drGobs2_cache = arrs;
	@unpack tworealconjhωHjₒjₛω_r₁r₂, tworealconjhωconjHjₒjₛω_r₂r₁ = arrs;
	tworealconjhωconjHjₒjₛω_r₂r₁P, tworealconjhωHjₒjₛω_r₁r₂P = map(parent, (tworealconjhωconjHjₒjₛω_r₂r₁, tworealconjhωHjₒjₛω_r₁r₂));

	HT = HybridArray{Tuple{sizeindG(los)...,sizeindG(los)...,StaticArrays.Dynamic()}}
	H₂₁ = _structarrayparent(Hjₒjₛω_r₂r₁, HT)
	H₁₂ = _structarrayparent(Hjₒjₛω_r₁r₂, HT)

	tworealconjhωconjHjₒjₛω_r₂r₁S = reinterpretSMatrix(tworealconjhωconjHjₒjₛω_r₂r₁P)
	tworealconjhωHjₒjₛω_r₁r₂S = reinterpretSMatrix(tworealconjhωHjₒjₛω_r₁r₂P)

	ℓ_range_proc = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)

	jₒjₛ_allmodes = L2L1Triangle(ℓ_range_proc, s_max, ℓ_arr)
	jₒrange = l2_range(jₒjₛ_allmodes)
	@unpack Gobs1_cache, Gobs2_cache = arrs
	for jₒ in jₒrange
		Gobs1_cache[jₒ] = zeros(ComplexF64, axes(Gsrc))
		drGobs1_cache[jₒ] = zeros(ComplexF64, axes(drGsrc))
		if !obs_at_same_height
			Gobs2_cache[jₒ] = zeros(ComplexF64, axes(Gsrc))
			drGobs2_cache[jₒ] = zeros(ComplexF64, axes(drGsrc))
		end
	end

	@timeit localtimer "CG" begin
		C = SHVector([zeros(abs(jₒ-jₛ):jₒ+jₛ) for (jₒ, jₛ) in jₒjₛ_allmodes], jₒjₛ_allmodes)
		for (jₒ, jₛ) in jₒjₛ_allmodes
			C_jₒjₛ = C[(jₒ, jₛ)]
			for l in abs(jₒ-jₛ):jₒ+jₛ
				if isodd(jₒ+jₛ+l)
					continue
				end
				C_jₒjₛ[l] = clebschgordan(Float64, jₒ, 0, jₛ, 0, l, 0)
			end
		end
	end

	_K = zeros(nr, length(SHModes))
	K = StructArray{ComplexF64}((_K, zero(_K)))

	@timeit localtimer "biposh" begin
		Y12, Y21 = los_projected_spheroidal_biposh_flippoints(xobs1, xobs2, los, SHModes, jₒjₛ_allmodes)
	end

	ω_ind_prev = -1
	# Loop over the Greenfn files
	for (jₛ, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]
		conjhω = conj(hω_arr[ω_ind])
		dωω²Pω = dω/2π * ω^2 * Powspec(ω)

		@timeit localtimer "FITS" begin

		# Green function about the source radius
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (jₛ, ω_ind), NGfn_files_src, 1:2, srcindFITS(los), :, 1)

		Gαr_r₁_rsrc = αrcomp(Gsrc, r₁_ind, los)
		Gαr_r₂_rsrc = αrcomp(Gsrc, r₂_ind, los)

		# Derivative of Green function about the source radius
		read_Gfn_file_at_index!(drGsrc, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (jₛ, ω_ind), NGfn_files_src, 1:1, srcindFITS(los), :, 2)
		end #localtimer

		for jₒ in l2_range(jₒjₛ_allmodes, jₛ)

			jₒjₛ_ind = modeindex(jₒjₛ_allmodes, (jₒ, jₛ))

			Yjₒjₛ_lm_n₁n₂ = Y12[jₒjₛ_ind]
			Yjₒjₛ_lm_n₂n₁ = Y21[jₒjₛ_ind]

			(Yjₒjₛ_lm_n₁n₂ === nothing || Yjₒjₛ_lm_n₂n₁ === nothing) && continue

			Cljₒjₛ = C[(jₒ, jₛ)]

			Gobs1 = Gobs1_cache[jₒ]
			drGobs1 = drGobs1_cache[jₒ]
			Gobs2 = Gobs2_cache[jₒ]
			drGobs2 = drGobs2_cache[jₒ]

			@timeit localtimer "FITS" begin

			if ω_ind != ω_ind_prev || jₒ ∉ l2_range(jₒjₛ_allmodes, jₛ-1)
				# Green functions based at the observation point for ℓ′
				read_Gfn_file_at_index!(Gobs1, Gfn_fits_files_obs1,
				(ℓ_arr, 1:Nν_Gfn), (jₒ, ω_ind), NGfn_files_obs1, 1:2, srcindFITS(los), :, 1)

				read_Gfn_file_at_index!(drGobs1, Gfn_fits_files_obs1,
				(ℓ_arr, 1:Nν_Gfn), (jₒ, ω_ind), NGfn_files_obs1, 1:1, srcindFITS(los), :, 2)
			end

			end # timer

			@timeit localtimer "radial term" begin

			# precompute the radial term in f
			radial_fn_δc_firstborn!(fjₒjₛ_r₁_rsrc, Gsrc, drGsrc, jₛ, divGsrc,
				Gobs1, drGobs1, jₒ, divGobs)

			end # timer

			@timeit localtimer "radial term 2" begin
				Hjₒjₛω!(Hjₒjₛω_r₁r₂, fjₒjₛ_r₁_rsrc, Gαr_r₂_rsrc)
			end
			@timeit localtimer "radial term 3" begin
				@turbo for I in eachindex(H₁₂)
					tworealconjhωHjₒjₛω_r₁r₂P[I] = 2real(conjhω * H₁₂[I])
				end
			end

			if !obs_at_same_height

				@timeit localtimer "FITS" begin
				if ω_ind != ω_ind_prev || jₒ ∉ l2_range(jₒjₛ_allmodes, jₛ-1)
					read_Gfn_file_at_index!(Gobs2, Gfn_fits_files_obs2,
					(ℓ_arr, 1:Nν_Gfn), (jₒ, ω_ind), NGfn_files_obs2, 1:2, srcindFITS(los), :, 1)

					read_Gfn_file_at_index!(drGobs2, Gfn_fits_files_obs2,
					(ℓ_arr, 1:Nν_Gfn), (jₒ, ω_ind), NGfn_files_obs2, 1:1, srcindFITS(los), :, 2)
				end
				end # timer

				@timeit localtimer "radial term" begin
				radial_fn_δc_firstborn!(fjₒjₛ_r₂_rsrc, Gsrc, drGsrc, jₛ, divGsrc,
					Gobs2, drGobs2, jₒ, divGobs)
				end # timer

				@timeit localtimer "radial term 2" begin
				Hjₒjₛω!(Hjₒjₛω_r₂r₁, fjₒjₛ_r₂_rsrc, Gαr_r₁_rsrc)
				end
			end
			@timeit localtimer "radial term 3" begin
				@turbo for I in eachindex(H₂₁)
					tworealconjhωconjHjₒjₛω_r₂r₁P[I] = 2real(conjhω * conj(H₂₁[I]))
				end
			end

			for (l, m) in firstshmodes(Yjₒjₛ_lm_n₁n₂)

				# The Clebsch-Gordan coefficients imply the selection
				# rule that only even ℓ+ℓ′+s modes contribute
				isodd(jₒ + jₛ + l) && continue

				# Check if triangle condition is satisfied
				# The loop is over all possible ℓ and ℓ′ for a given s_max
				# Not all combinations would contribute towards a specific s
				δ(jₛ, jₒ, l) || continue

				pre = dωω²Pω * Njₒjₛs(jₒ, jₛ, l) * Cljₒjₛ[l]

				@timeit localtimer "kernel" begin

				populatekernel!(SoundSpeed(), los, K, SHModes,
					Yjₒjₛ_lm_n₁n₂, Yjₒjₛ_lm_n₂n₁,
					(l, m), (jₛ, jₒ, ω), pre,
					tworealconjhωHjₒjₛω_r₁r₂S,
					tworealconjhωconjHjₒjₛω_r₂r₁S)
				end
			end # lm
		end # jₒ

		ω_ind_prev = ω_ind
	end # (jₛ, ω)

	map(closeGfnfits, (Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2))
	Array(K)
end

function kernelfilenameδcₛₜ(m::SeismicMeasurement, los::los_direction, j_range, SHModes)
	mtag = m isa TravelTimes ? "δτ" : "A"
	lostag = los isa los_radial ? "" : "_los"
	"Kst_$(mtag)_c_$(modetag(j_range, SHModes))$(lostag).fits"
end

function kernel_δcₛₜ(comm, m::SeismicMeasurement, xobs1, xobs2,
	los::los_direction = los_radial(); kwargs...)

	SHModes = getkernelmodes(; kwargs...)

	r_src, = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc, p_Gobs1, p_Gobs2 = read_parameters_for_points(xobs1, xobs2; kwargs..., c_scale = 1)
	@unpack ν_arr, ℓ_arr = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr,ν_arr; kwargs...)
	j_range, ν_ind_range = iters

	hω_arr = get(kwargs, :hω) do
		h = hω(comm, m, xobs1, xobs2, los; kwargs..., print_timings = false)
		_broadcast(h, 0, comm)
	end

	K_δcₛₜ = pmapsum(comm, kernel_δcₛₜ_partial, iters,
		xobs1, xobs2, los, SHModes, hω_arr, p_Gsrc, p_Gobs1, p_Gobs2, r_src)

	K_δcₛₜ === nothing && return nothing

	if get(kwargs, :save, true)
		filename = joinpath(SCRATCH_kerneldir[], kernelfilenameδcₛₜ(m, los, j_range, SHModes))
		header = generatefitsheader(xobs1, xobs2, SHModes, j_range, ν_ind_range)
		FITS(filename,"w") do f
			FITSIO.write(f, reinterpret_as_float(K_δcₛₜ), header = header)
		end
	end

	SHArray(K_δcₛₜ, (axes(K_δcₛₜ, 1), SHModes))
end
