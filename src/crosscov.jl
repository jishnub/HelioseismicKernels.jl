########################################################################################
# Macro to call appropriate 3D method given a 2D one
########################################################################################

macro two_points_on_the_surface(fn)
	callermodule = __module__
	quote
		function $(esc(fn))(comm, nobs1::Point2D, nobs2::Point2D,
			los::los_direction = los_radial(); kwargs...)

			r_obs = get(kwargs, :r_obs, r_obs_default)
			xobs1 = Point3D(r_obs, nobs1)
			xobs2 = Point3D(r_obs, nobs2)
			$callermodule.$fn(comm, xobs1, xobs2, los; kwargs...)
		end
	end
end

########################################################################################
# Get the modes to be used
########################################################################################

function ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)
	ℓ_range = get(kwargs, :ℓ_range, ℓ_arr)
	ν_ind_range = get(kwargs, :ν_ind_range, axes(ν_arr, 1))
	ℓ_range, ν_ind_range
end

#######################################################################################################
# Full Frequency axis
#######################################################################################################

function pad_zeros_ν(arr::Array, ν_ind_range, Nν_Gfn, ν_start_zeros, ν_end_zeros, dim::Integer = 1)
 	ax_leading  = axes(arr)[1:dim-1]
	ax_trailing = axes(arr)[dim+1:end]
	N_leading_zeros_ν = ν_start_zeros + first(ν_ind_range) - 1
	N_trailing_zeros_ν = Nν_Gfn - last(ν_ind_range) + ν_end_zeros

	inds_leading = (Tuple(ax_leading)..., oftype(axes(arr,1), 1:N_leading_zeros_ν), Tuple(ax_trailing)...)
	inds_trailing = (Tuple(ax_leading)..., oftype(axes(arr,1), 1:N_trailing_zeros_ν) , Tuple(ax_trailing)...)

	lead_arr = zeros(eltype(arr), inds_leading)
	trail_arr = zeros(eltype(arr), inds_trailing)

	cat(lead_arr, arr, trail_arr, dims = dim)
end

# If an OffsetArray is passed we may infer the frequency range from its axes
function pad_zeros_ν(arr::OffsetArray, Nν_Gfn,
	ν_start_zeros, ν_end_zeros, dim::Integer = 1)

	pad_zeros_ν(parent(arr), axes(arr, dim), Nν_Gfn, ν_start_zeros, ν_end_zeros, dim)
end

########################################################################################
# Read parameters for source and observation points
########################################################################################

read_parameters_for_points(; kwargs...) = read_all_parameters(; kwargs...)

function read_parameters_for_points(xobs1::Point3D, xobs2::Point3D; kwargs...)
	p_Gsrc = read_all_parameters(; kwargs...)
	p_Gobs1 = read_all_parameters(; kwargs..., r_src = xobs1.r)
	p_Gobs2 = read_all_parameters(; kwargs..., r_src = xobs2.r)
	return p_Gsrc, p_Gobs1, p_Gobs2
end

function read_parameters_for_points(xobs1::Point3D, xobs2_arr::Vector{<:Point3D}; kwargs...)
	p_Gsrc = read_all_parameters(; kwargs...)
	p_Gobs1 = read_all_parameters(; kwargs..., r_src = xobs1.r)
	p_Gobs2 = [read_all_parameters(; kwargs..., r_src = xobs2.r) for xobs2 in xobs2_arr]
	return p_Gsrc, p_Gobs1, p_Gobs2
end

function read_parameters_for_points(::Point2D, ::Union{Point2D, Vector{<:Point2D}}; kwargs...)
	p_Gsrc = read_all_parameters(; kwargs...)
	r_obs = get(kwargs, :r_obs, r_obs_default)
	p_Gobs = read_all_parameters(; kwargs..., r_src = r_obs)
	return p_Gsrc, p_Gobs, p_Gobs
end

function read_parameters_for_points(::Point2D, ::Vector{<:Point2D}; kwargs...)
	p_Gsrc = read_all_parameters(; kwargs...)
	r_obs = get(kwargs, :r_obs, r_obs_default)
	p_Gobs = read_all_parameters(; kwargs..., r_src = r_obs)
	return p_Gsrc, p_Gobs, p_Gobs
end

########################################################################################
# Bipolar harmonics
########################################################################################

function computeY₀₀(::los_radial, xobs1::SphericalPoint, xobs2::SphericalPoint, ℓ_range::AbstractRange{<:Integer})
	v = collectPl(cosχ(xobs1, xobs2), lmax = maximum(ℓ_range))[ℓ_range]
	norm(ℓ) = (2ℓ+1)/4π
	@. v *= norm(ℓ_range)
	return OffsetArray(v, ℓ_range)
end

function computeY₀₀(::los_earth, xobs1::SphericalPoint, xobs2::SphericalPoint, ℓ_range::AbstractRange{<:Integer})
	B = BipolarSphericalHarmonics.monopolarharmonics(GSH(), Point2D(xobs1)..., Point2D(xobs2)..., maximum(ℓ_range))
	v = [biposh(B, Point2D(xobs1)..., Point2D(xobs2)..., 0, 0, ℓ, ℓ) for ℓ in ℓ_range]
	norm(ℓ) = (-1)^ℓ * √(2ℓ+1)
	@. v *= norm(ℓ_range)
	OffsetArray(v, ℓ_range)
end

function computeY₁₀(::los_radial, xobs1::SphericalPoint, xobs2::SphericalPoint, ℓ_range::AbstractRange{<:Integer})
	v = epsilon.(collectPl(cosχ(xobs1, Point2D(xobs2.θ, Dual(xobs2.ϕ, 1))), lmax = maximum(ℓ_range)))[ℓ_range]
	norm(ℓ) = (2ℓ+1)/4π
	@. v *= norm(ℓ_range)
	return OffsetArray(v, ℓ_range)
end

function computeY₁₀(::los_earth, xobs1::SphericalPoint, xobs2::SphericalPoint, ℓ_range::AbstractRange{<:Integer})
	B = BipolarSphericalHarmonics.monopolarharmonics(GSH(), Point2D(xobs1)..., Point2D(xobs2)..., maximum(ℓ_range))
	v = [biposh(B, Point2D(xobs1)..., Point2D(xobs2)..., 1, 0, ℓ, ℓ) for ℓ in ℓ_range]
	norm(ℓ) = (-1)^ℓ * √(2ℓ+1)
	@. v *= norm(ℓ_range)
	OffsetArray(v, ℓ_range)
end

los_projected_biposh(::los_radial, Y1, l1, l2) = Y1
function _multiplylos!(Y12, j1j2ind, Y12_j1j2::AbstractVector, l1l2::AbstractVector)
	for (lmind, Y12_j1j2_lm) in pairs(Y12_j1j2)
		Y12_j1j2[lmind] = Y12_j1j2_lm .* l1l2
	end
	nothing
end
function _multiplylos!(Y12, j1j2ind, Y12_j1j2::StaticVector, l1l2::AbstractVector)
	Y12[j1j2ind] = Y12_j1j2 .* l1l2
	nothing
end
function los_projected_biposh(::los_earth, Y12, l1, l2)
	l1l2 = LinearAlgebra.kron(parent(l1), parent(l2))
	for (j1j2ind, Y12_j1j2) in pairs(Y12)
		Y12_j1j2 === nothing && continue
		_multiplylos!(Y12, j1j2ind, Y12_j1j2, l1l2)
	end
	return Y12
end

biposh_spheroidal(Y::OffsetVector) =
	OffsetArray(biposh_spheroidal(parent(Y)), axes(Y))
function biposh_spheroidal(Y::SHVector)
	SHArray(biposh_spheroidal(parent(Y)), SphericalHarmonicArrays.modes(Y))
end
biposh_spheroidal(Y::AbstractVector{<:Number}) = Y
biposh_spheroidal(Y::Number) = Y
biposh_spheroidal(Y::Vector) = [biposh_spheroidal(Yi) for Yi in Y]
function biposh_spheroidal(Y::SVector{9,<:Number})
	T00 = Y[kronindex(GSH(), 0, 0)]
	T10 = Y[kronindex(GSH(), 1, 0)] + Y[kronindex(GSH(), -1, 0)]
	T01 = Y[kronindex(GSH(), 0, 1)] + Y[kronindex(GSH(), 0, -1)]
	T11 = Y[kronindex(GSH(), -1, -1)] + Y[kronindex(GSH(), 1, -1)] +
			Y[kronindex(GSH(), -1, 1)] + Y[kronindex(GSH(), 1, 1)]
	OffsetArray(SMatrix{2,2}(T00, T10, T01, T11), 0:1, 0:1)
end

function los_projected_biposh_spheroidal(computeY, xobs1, xobs2::SphericalPoint, los, ℓ_range)
	_Y12 = computeY(los, xobs1, xobs2, ℓ_range)
	l1, l2 = line_of_sight_covariant.((xobs1, xobs2), los)
	biposh_spheroidal(los_projected_biposh(los, _Y12, l1, l2))
end
function los_projected_biposh_spheroidal(computeY, nobs1, nobs2_arr::Vector{<:SphericalPoint}, los, ℓ_range)
	l1 = line_of_sight_covariant(nobs1, los)
	l2arr = line_of_sight_covariant.(nobs2_arr, los)
	_Y12 = computeY.(los, nobs1, nobs2_arr, (ℓ_range,))
	OffsetArray(permutedims(reduce(hcat,
	biposh_spheroidal.(los_projected_biposh.(los, _Y12, (l1,), l2arr)))), :, ℓ_range)
end

########################################################################################

function allocateGfn(los::los_direction, obs_at_same_height::Bool)
	α_r₁ = zeros(ComplexF64, obsindG(los))
	α_r₂ = obs_at_same_height ? α_r₁ : zeros(ComplexF64, obsindG(los))
	(; α_r₁, α_r₂)
end

function C_FITS_header(xobs1::Point3D, xobs2::Point3D)
	FITSHeader(["X1R", "X1THETA", "X1PHI", "X2R", "X2THETA", "X2PHI"],
		Any[float(xobs1.r), float(xobs1.θ), float(xobs1.ϕ),
		float(xobs2.r), float(xobs2.θ), float(xobs2.ϕ)],
		["Radius of the first observation point",
		"Co-latitude of the first observation point",
		"Azimuth of the first observation point",
		"Radius of the second observation point",
		"Co-latitude of the second observation point",
		"Azimuth of the second observation point"])
end

function C_FITS_header(n1::Point2D, n2::Point2D)
	xobs1 = Point3D(r_obs_default, n1)
	xobs2 = Point3D(r_obs_default, n2)
	C_FITS_header(xobs1, xobs2)
end

function C_FITS_header(n1::Point2D, n2::Vector{<:Point2D})
	xobs1 = Point3D(r_obs_default, n1)
	xobs2 = Point3D(r_obs_default, n2[1])
	header = C_FITS_header(xobs1, xobs2)

	θmin = minimum(x->float(x.θ), n2)
	header["THMIN"] = θmin
	θmax = maximum(x->float(x.θ), n2)
	header["THMAX"] = θmax

	ϕmin = minimum(x->float(x.ϕ), n2)
	header["PHIMIN"] = ϕmin
	ϕmax = maximum(x->float(x.ϕ), n2)
	header["PHIMAX"] = ϕmax

	header["NTHPHI"] = length(n2)

	header
end

########################################################################################
# cross-covariances
########################################################################################

function Cωℓ(::los_radial, ω, ℓ, α_r₁::AbstractArray{<:Any, 0}, α_r₂::AbstractArray{<:Any, 0}, Pl)
	ω^2 * Powspec(ω) * conj(α_r₁[]) * α_r₂[] * Pl
end

function Cωℓ(::los_earth, ω, ℓ, α_r₁::AbstractVector{<:Complex}, α_r₂::AbstractVector{<:Complex}, Y12)
	pre = ω^2 * Powspec(ω)
	s = zero(ComplexF64)
	for α in 0:1, β in 0:1
		s += pre * conj(α_r₁[α]) * α_r₂[β] * Y12[α, β]
	end
	s
end

function Cω_partial(localtimer, ℓ_ωind_iter_on_proc, xobs1::Point3D, xobs2::Point3D, los::los_direction,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, r_obs = nothing, c_scale = 1)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = c_scale)
	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src, (ℓ_arr, 1:Nν_Gfn),
		ℓ_ωind_iter_on_proc, NGfn_files_src)

	ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)[end]

	Cω_proc = zeros(ComplexF64, ν_ind_range)

	ℓ_range = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)

	Y12 = los_projected_biposh_spheroidal(computeY₀₀, xobs1, xobs2, los, ℓ_range)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	@unpack α_r₁, α_r₂ = allocateGfn(los, r₁_ind == r₂_ind)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]

		@timeit localtimer "FITS" begin
			read_Gfn_file_at_index!(α_r₁, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₁_ind, obsindFITS(los), 1, 1)

			if r₁_ind != r₂_ind
				read_Gfn_file_at_index!(α_r₂, Gfn_fits_files_src,
				(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₂_ind, obsindFITS(los), 1, 1)
			end
		end

		@timeit localtimer "Cω calculation" begin
			Cω_proc[ω_ind] += Cωℓ(los, ω, ℓ, α_r₁, α_r₂, Y12[ℓ])
		end
	end
	closeGfnfits(Gfn_fits_files_src)
	parent(Cω_proc)
end

function Cω_partial(localtimer, ℓ_ωind_iter_on_proc,
	nobs1::Point2D, nobs2_arr::Vector{<:Point2D}, los::los_direction,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, r_obs = r_obs_default, c_scale = 1)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = c_scale)

	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)[end]

	r₁_ind = radial_grid_index(r_obs)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
		(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, NGfn_files_src)

	@unpack α_r₁ = allocateGfn(los, true)

	Cω_proc = zeros(ComplexF64, length(nobs2_arr), ν_ind_range)

	ℓ_range = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)
	Y12 = los_projected_biposh_spheroidal(computeY₀₀, nobs1, nobs2_arr, los, ℓ_range)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]

		read_Gfn_file_at_index!(α_r₁, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₁_ind, obsindFITS(los), 1, 1)

		for n2ind in axes(Cω_proc, 1)
			Cω_proc[n2ind, ω_ind] += Cωℓ(los, ω, ℓ, α_r₁, α_r₁, Y12[n2ind, ℓ])
		end
	end
	closeGfnfits(Gfn_fits_files_src)
	permutedims(parent(Cω_proc))
end

########################################################################################################
# Derivatives of cross-covariance (useful in the radial case)
########################################################################################################

function ∂ϕ₂Cω_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D, xobs2::Point3D, los::los_radial,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, r_obs = nothing, c_scale = 1)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = c_scale)
	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	_, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, NGfn_files_src)

	Cω_proc = zeros(ComplexF64, ν_ind_range)

	ℓ_range = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)
	Y12 = los_projected_biposh_spheroidal(computeY₁₀, xobs1, xobs2, los, ℓ_range)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	@unpack α_r₁, α_r₂ = allocateGfn(los, r₁_ind == r₂_ind)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		read_Gfn_file_at_index!(α_r₁, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₁_ind, obsindFITS(los), 1, 1)

		if r₁_ind != r₂_ind
			read_Gfn_file_at_index!(α_r₂, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₂_ind, obsindFITS(los), 1, 1)
		end

		Cω_proc[ω_ind] += Cωℓ(los, ω, ℓ, α_r₁, α_r₂, Y12[ℓ])
	end
	closeGfnfits(Gfn_fits_files_src)
	parent(Cω_proc)
end

function ∂ϕ₂Cω_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	nobs1::Point2D, nobs2_arr::Vector{<:Point2D}, los::los_radial,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, r_obs = nothing, c_scale = 1)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = c_scale)
	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, Nν_Gfn = p_Gsrc

	ℓ_arr, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, NGfn_files_src)

	Cω_proc = zeros(ComplexF64, ν_ind_range)

	ℓ_range = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)
	Y12 = los_projected_biposh_spheroidal(computeY₁₀, nobs1, nobs2_arr, los, ℓ_range)

	r_obs_ind = radial_grid_index(r_obs)

	@unpack α_r₁, α_r₂ = allocateGfn(los, true)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		read_Gfn_file_at_index!(α_r₁, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r_obs_ind, obsindFITS(los), 1, 1)

		for n2ind in eachindex(nobs2_arr)
			Cω_proc[n2ind, ω_ind] += Cωℓ(los, ω, ℓ, α_r₁, α_r₂, Y12[n2ind, ℓ])
		end
	end
	closeGfnfits(Gfn_fits_files_src)
	permutedims(parent(Cω_proc))
end

function Cω_∂ϕ₂Cω_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D, xobs2::Point3D, los::los_radial,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, r_obs = nothing, c_scale = 1)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = c_scale)

	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	_, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, NGfn_files_src)

	Cω_proc = zeros(ComplexF64, 0:1, ν_ind_range)

	ℓ_range = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)
	Pl_cosχ = los_projected_biposh_spheroidal(computeY₀₀, xobs1, xobs2, los, ℓ_range)
	∂ϕ₂Pl_cosχ = los_projected_biposh_spheroidal(computeY₁₀, xobs1, xobs2, los, ℓ_range)

	@unpack α_r₁, α_r₂ = allocateGfn(los, r₁_ind == r₂_ind)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		read_Gfn_file_at_index!(α_r₁, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₁_ind, obsindFITS(los), 1, 1)
		if r₁_ind != r₂_ind
			read_Gfn_file_at_index!(α_r₂, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₂_ind, obsindFITS(los), 1, 1)
		end

		f = Cωℓ(los, ω, ℓ, α_r₁, α_r₂, 1)

		Cω_proc[0, ω_ind] += f * Pl_cosχ[ℓ]
		Cω_proc[1, ω_ind] += f * ∂ϕ₂Pl_cosχ[ℓ]
	end
	closeGfnfits(Gfn_fits_files_src)
	permutedims(parent(Cω_proc))
end

function Cω_∂ϕ₂Cω_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	nobs1::Point2D, nobs2_arr::Vector{<:Point2D}, los::los_radial,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, r_obs = r_obs_default, c_scale = 1)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = c_scale)
	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	_, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,
						(ℓ_arr, 1:Nν_Gfn), ℓ_ωind_iter_on_proc, NGfn_files_src)

	Cω_proc = zeros(ComplexF64, 0:1, length(nobs2_arr), ν_ind_range)

	r_obs_ind = radial_grid_index(r_obs)

	lmax = maximum(ℓ_arr)

	∂ϕ₂Pl_cosχ = zeros(0:lmax, length(nobs2_arr))
	Pl_cosχ = zeros(0:lmax, length(nobs2_arr))

	ℓ_range = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)
	Pl_cosχ = los_projected_biposh_spheroidal(computeY₀₀, nobs1, nobs2_arr, los, ℓ_range)
	∂ϕ₂Pl_cosχ = los_projected_biposh_spheroidal(computeY₁₀, nobs1, nobs2_arr, los, ℓ_range)

	@unpack α_r₁ = allocateGfn(los, true)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		read_Gfn_file_at_index!(α_r₁, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r_obs_ind, obsindFITS(los), 1, 1)

		f = Cωℓ(los_radial(), ω, ℓ, α_r₁, α_r₁, 1)

		for n2ind in axes(Cω_proc, 2)
			Cω_proc[0, n2ind, ω_ind] += f * Pl_cosχ[n2ind, ℓ]
			Cω_proc[1, n2ind, ω_ind] += f * ∂ϕ₂Pl_cosχ[n2ind, ℓ]
		end
	end
	closeGfnfits(Gfn_fits_files_src)
	permutedims(parent(Cω_proc),[3, 2, 1])
end

########################################################################################
# Functions that iterate over the modes in parallel
########################################################################################

for f in (:Cω, :∂ϕ₂Cω, :Cω_∂ϕ₂Cω)
	f_str = String(f)
	f_feeder = Symbol(f, "_feeder")
	f_partial = Symbol(f, "_partial")
	@eval function $f_feeder(comm, args...; kwargs...)
		r_src, r_obs, c_scale = read_rsrc_robs_c_scale(kwargs)
		p_Gsrc = read_all_parameters(r_src = r_src, c_scale = c_scale)
		@unpack Nν_Gfn, ν_arr, ℓ_arr, ν_start_zeros, ν_end_zeros = p_Gsrc
		iters = ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)

		C = pmapsum(comm, $f_partial, iters, args..., p_Gsrc, r_src, r_obs, c_scale)
		C === nothing && return nothing
		ν_ind_range = last(iters)
		return pad_zeros_ν(C, ν_ind_range, Nν_Gfn, ν_start_zeros, ν_end_zeros)
	end
	# With or without los, 3D points
	@eval function $f(comm, xobs1::Point3D, xobs2::Point3D, los::los_direction = los_radial(); kwargs...)
		C = $f_feeder(comm, xobs1, xobs2, los; kwargs...)
		C === nothing && return nothing

		lostag = los isa los_radial ? "" : "_los"
		filename = $f_str * lostag * ".fits"

		save_to_fits_and_return(filename, C, header = C_FITS_header(xobs1, xobs2))
	end
	# Multiple 2D points
	@eval function $f(comm, nobs1::Point2D, nobs2_arr::Vector{<:Point2D},
		los::los_direction = los_radial(); kwargs...)

		C = $f_feeder(comm, nobs1, nobs2_arr, los; kwargs...)
		C === nothing && return nothing
		save_to_fits_and_return($f_str * "_n2arr.fits", C, header = C_FITS_header(nobs1, nobs2_arr))
	end
	# With or without los, 2D points
	@eval @two_points_on_the_surface $f
end

########################################################################################################
# Spectrum of C(ℓ, ω)
########################################################################################################

function Cωℓ_spectrum_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	r_obs, los::los_radial,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src)
	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, Nν_Gfn, ω_arr = p_Gsrc

	ℓ_range, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src, (ℓ_arr, 1:Nν_Gfn),
						ℓ_ωind_iter_on_proc, NGfn_files_src)

	@unpack α_r₁, = allocateGfn(los, true)
	r₁_ind = radial_grid_index(r_obs)

	Cℓω = zeros(ℓ_range, ν_ind_range)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		# get the only element of a 0-dim array
		read_Gfn_file_at_index!(α_r₁, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₁_ind, obsindFITS(los), 1, 1)

		# m-averaged, so divided by 2ℓ+1
		Cℓω[ℓ, ω_ind] = ω^2 * Powspec(ω) * 1/4π * abs2(α_r₁[])
	end
	closeGfnfits(Gfn_fits_files_src)
	permutedims(parent(Cℓω))
end

function Cωℓ_spectrum(comm; kwargs...)
	r_src, r_obs, c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(r_src = r_src, c_scale = c_scale)
	@unpack ℓ_arr, ν_arr = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)

	Cωℓ_in_range = pmapsum(comm, Cωℓ_spectrum_partial, iters,
		r_obs, los_radial(), p_Gsrc, r_src)

	Cωℓ_in_range === nothing && return nothing
	save_to_fits_and_return("Cωℓ_in_range.fits", Cωℓ_in_range)
end

########################################################################################################
# Time-domain cross-covariance
########################################################################################################

for (_fnω, _fnt) in ((:Cω, :Ct), (:∂ϕ₂Cω, :∂ϕ₂Ct))
	_fnts = String(_fnt)
	@eval function $_fnt(comm, xobs1, xobs2, los::los_direction = los_radial(); kwargs...)
		C = $_fnω(comm, xobs1, xobs2, los; kwargs...)
		C === nothing && return nothing

		C_t = $_fnt(C; kwargs...)
		lostag = los isa los_radial ? "" : "_los"
		vtag = xobs2 isa AbstractVector ? "_n2arr" : ""
		filename = $_fnts * vtag * lostag * ".fits"
		save_to_fits_and_return(filename, C_t, header = C_FITS_header(xobs1, xobs2))
	end
	@eval function $_fnt(Cω_arr::AbstractArray{<:Complex}; kwargs...)
		dν = get(kwargs, :dν) do
			read_parameters("dν"; kwargs...)[1]
		end

		τ_ind_range = get(kwargs, :τ_ind_range, Colon())

		C_t = fft_ω_to_t(Cω_arr, dν)

		if !(τ_ind_range isa Colon)
			C_t = C_t[τ_ind_range,..]
		end

		return C_t
	end
end

########################################################################################################
# Cross-covariance at all distances on the equator, essentially the time-distance diagram
########################################################################################################

function CΔϕω_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	r₁, r₂, los::los_radial, Δϕ_arr,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, c_scale = 1)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = c_scale)

	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	ℓ_arr, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	r₁_ind, r₂_ind = radial_grid_index.((r₁, r₂))

	nϕ = length(Δϕ_arr)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src, (ℓ_arr, 1:Nν_Gfn),
						ℓ_ωind_iter_on_proc, NGfn_files_src)

	Cϕω_arr = zeros(ComplexF64, nϕ, ν_ind_range)

	Pl_cosχ = permutedims(computeY₀₀.(los, Point2D(pi/2, 0), Point2D.(pi/2, Δϕ_arr), (ℓ_arr,)))

	@unpack α_r₁, α_r₂ = allocateGfn(los, r₁_ind == r₂_ind)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		read_Gfn_file_at_index!(α_r₁, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₁_ind, obsindFITS(los), 1, 1)
		if r₁_ind != r₂_ind
			read_Gfn_file_at_index!(α_r₂, Gfn_fits_files_src,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₂_ind, obsindFITS(los), 1, 1)
		end

		for ϕ_ind in 1:nϕ
			Cϕω_arr[ϕ_ind, ω_ind] += Cωℓ(los_radial(), ω, ℓ, α_r₁, α_r₂, Pl_cosχ[ϕ_ind, ℓ])
		end
	end
	closeGfnfits(Gfn_fits_files_src)
	parent(Cϕω_arr)
end

function CΔϕω_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D, r₂, ::los_earth, Δϕ_arr, p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, c_scale = 1)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = c_scale)

	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, Nν_Gfn = p_Gsrc

	_, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, r₂))

	nϕ = length(Δϕ_arr)

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src, (ℓ_arr, 1:Nν_Gfn),
						ℓ_ωind_iter_on_proc, NGfn_files_src)

	Cϕω_arr = zeros(ComplexF64, nϕ, ν_ind_range)

	# covariant components
	ℓ_ωind_iter_on_proc = sort(collect(ℓ_ωind_iter_on_proc))

	xobs2_arr = [Point3D(r₂, xobs1.θ, xobs1.ϕ + Δϕ) for Δϕ in Δϕ_arr]

	_Y12 = computeY₀₀.(los, nobs1, nobs2_arr, (ℓ_range,))

	# covariant components
	l1 = line_of_sight_covariant(xobs1, los)
	l2 = line_of_sight_covariant.(xobs2_arr, los)

	Y12 = OffsetArray(permutedims(reduce(hcat,
	biposh_spheroidal.(los_projected_biposh.(los, _Y12, (l1,), l2)))), :, ℓ_range)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc
		ω = ω_arr[ω_ind]

		α_r₁ = read_Gfn_file_at_index(Gfn_fits_files_src,
			ℓ_arr, 1:Nν_Gfn, (ℓ, ω_ind), NGfn_files_src, r₁_ind, 1:2, 1, 1)
		α_r₂ = α_r₁
		if r₁_ind != r₂_ind
			α_r₂ = read_Gfn_file_at_index(Gfn_fits_files_src,
			ℓ_arr, 1:Nν_Gfn, (ℓ, ω_ind), NGfn_files_src, r₂_ind, 1:2, 1, 1)
		end


		for (ϕ_ind, ϕ) in enumerate(Δϕ_arr)
			Y12ℓ = Y12[ϕ_ind, ℓ]
			for β in 0:1, α in 0:1
				Cϕω_arr[ϕ_ind, ω_ind] += ω^2 * Powspec(ω) *
										conj(α_r₁[α]) * α_r₂[β] * Y12ℓ[α, β]
			end
		end
	end
	closeGfnfits(Gfn_fits_files_src)
	parent(Cϕω_arr)
end

function _CΔϕω(comm, args...; kwargs...)
	r_src, _, c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(; kwargs...)
	@unpack Nν_Gfn, ν_arr, ℓ_arr, ν_start_zeros, ν_end_zeros = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)
	lmax = maximum(ℓ_range)

	Δϕ_arr = get(kwargs, :Δϕ_arr, LinRange(0, π, lmax+1)[1:end-1])

	Cω_in_range = pmapsum(comm, CΔϕω_partial, iters,
			args..., Δϕ_arr, p_Gsrc, r_src, c_scale)

	Cω_in_range === nothing && return nothing
	ν_ind_range = last(iters)
	return pad_zeros_ν(Cω_in_range, ν_ind_range, Nν_Gfn, ν_start_zeros, ν_end_zeros, 2)
end

function CΔϕω(comm, r₁ = r_obs_default, r₂ = r_obs_default,
	los::los_radial = los_radial(); kwargs...)
	CΔϕω = _CΔϕω(comm, r₁, r₂, los; kwargs...)
	CΔϕω === nothing && return nothing
	if get(kwargs, :save, true)
		filename = joinpath(SCRATCH_kerneldir[], "CΔϕω.fits")
		FITSIO.fitswrite(filename, CΔϕω)
	end
	return CΔϕω
end

function CtΔϕ(comm, r₁ = r_obs_default, r₂ = r_obs_default,
	los::los_direction = los_radial(); kwargs...)

	dν = get(kwargs, :dν) do
		read_parameters("dν"; kwargs...)[1]
	end

	C = CΔϕω(comm, r₁, r₂, los; kwargs..., save = false)
	C === nothing && return nothing
	CωΔϕ = permutedims(C)
	CtΔϕ = fft_ω_to_t(CωΔϕ, dν)

	τ_ind_range = get(kwargs, :τ_ind_range, Colon())
	if !(τ_ind_range isa Colon)
		CtΔϕ = CtΔϕ[τ_ind_range,..]
	end

	if get(kwargs, :save, true)
		tag = los isa los_radial ? "" : "_los"
		filename = "C_t_phi$tag.fits"
		FITSIO.fitswrite(filename, CtΔϕ)
	end
	return CtΔϕ
end

function CΔϕt(comm, r₁ = r_obs_default, r₂ = r_obs_default,
	los::los_direction = los_radial(); kwargs...)

	_Ctϕ = CtΔϕ(comm, r₁, r₂, los; kwargs..., save = false)
	_Ctϕ === nothing && return nothing
	C = permutedims(_Ctϕ)

	if get(kwargs, :save, true)
		tag = los isa los_radial ? "" : "_los"
		filename = "C_phi_t$tag.fits"
		FITSIO.fitswrite(filename, C)
	end
	return C
end

########################################################################################################
# Cross-covariance in a rotating frame
########################################################################################################

function Cτ_rotating_partial(localtimer, ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D, xobs2::Point3D, los::los_radial,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, c_scale = 1)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = c_scale)

	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	ℓ_range, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src, (ℓ_arr, 1:Nν_Gfn),
						ℓ_ωind_iter_on_proc, NGfn_files_src)

	α₁conjα₂ωℓ = zeros(ComplexF64, ℓ_range, ν_ind_range)

	α_r₁, α_r₂ = allocateGfn(los, r₁_ind == r₂_ind)

	# Read all radial parts
	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		read_Gfn_file_at_index!(α_r₁, Gfn_fits_files_src,
				(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₁_ind, 1, 1, 1)

		conjα₁_α₂ = complex(abs2(α_r₁[]))

		if r₁_ind != r₂_ind
			read_Gfn_file_at_index!(α_r₂, Gfn_fits_files_src,
				(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, r₂_ind, 1, 1, 1)

			conjα₁_α₂ = α_r₁[]*conj(α_r₂[])
		end

		α₁conjα₂ωℓ[ℓ, ω_ind] = conjα₁_α₂ * ω^2*Powspec(ω)
	end

	closeGfnfits(Gfn_fits_files_src)

	permutedims(parent(α₁conjα₂ωℓ))
end

# Without los, radial components, 3D points
function Cτ_rotating(comm, xobs1::Point3D, xobs2::Point3D, los::los_radial = los_radial(); kwargs...)
	# Return C(Δϕ, ω) = RFFT(C(Δϕ,τ)) = RFFT(IRFFT(C0(Δϕ - Ωτ, ω))(τ))
	r_src, _, c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(; kwargs...)
	@unpack Nν_Gfn, ν_arr, ℓ_arr, ν_start_zeros, ν_end_zeros, Nt, dt, dν = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)
	ℓ_range, ν_ind_range = iters

	Ω_rot = get(kwargs, :Ω_rot, 20e2/Rsun)
	τ_ind_range = get(kwargs, :τ_ind_range, 1:Nt)

	lmax = maximum(ℓ_range)

	Nτ = length(τ_ind_range)

	# The first step is loading in the αωℓ
	# Fourier transform the αωℓ and transpose to obtain αℓt
	Cω_in_range = pmapsum(comm, Cτ_rotating_partial, iters,
			xobs1, xobs2, los, p_Gsrc, r_src, c_scale)

	αωℓ = pad_zeros_ν(Cω_in_range, ν_ind_range, Nν_Gfn, ν_start_zeros, ν_end_zeros)

	αtℓ = fft_ω_to_t(αωℓ, dν)
	αℓt = permutedims(αtℓ)[:,τ_ind_range]

	np = min(nworkers(), Nτ)

	αℓt = distribute(αℓt, procs = workers()[1:np], dist=[1, np])

	τ_ind_tracker = RemoteChannel(()->Channel{Bool}(length(τ_ind_range)))
	progress_bar_τ = Progress(length(τ_ind_range), 1,"Cτ : ")

	Cτ_arr = @sync begin
		t = @async begin
			Cτ_arr =
			try
				DArray((Nτ,), workers()[1:np],[np]) do inds
					τ_ind_range_proc = τ_ind_range[first(inds)]
					αℓt_local = OffsetArray(αℓt[:lp], ℓ_range,τ_ind_range_proc)
					Cτ_arr = zeros(τ_ind_range_proc)
					for τ_ind in τ_ind_range_proc
						# τ goes from -T/2 to T/2-dt but is fftshifted
						τ = (τ_ind<=div(Nt, 2) ? (τ_ind-1) : (τ_ind-1- Nt)) * dt
						xobs2′ = Point3D(xobs2.r, xobs2.θ, xobs2.ϕ-Ω_rot*τ)
						Pl_cosχ = collectPl(cosχ(xobs1, xobs2′), lmax = lmax)
						for ℓ in ℓ_range
							Cτ_arr[τ_ind] += (2ℓ+1)/4π * αℓt_local[ℓ,τ_ind] * Pl_cosχ[ℓ]
						end
						put!(τ_ind_tracker, true)
					end
					finalize(τ_ind_tracker)
					parent(Cτ_arr)
				end
			finally
				put!(τ_ind_tracker, false)
			end
		end
		while take!(τ_ind_tracker)
			next!(progress_bar_τ)
		end
		fetch(t)
	end

	OffsetArray(Array(Cτ_arr), τ_ind_range)
end

# 2D points
@two_points_on_the_surface Cτ_rotating

# Without los, radial components, multiple 3D points
function Cτ_rotating(comm, xobs1, xobs2_arr::Vector{T}, los::los_direction = los_radial();
	Ω_rot = 20e2/Rsun,τ_ind_range = nothing, kwargs...) where {T<:SphericalPoint}

	# Return C(Δϕ, ω) = RFFT(C(Δϕ,τ)) = RFFT(IRFFT(C0(Δϕ - Ωτ, ω))(τ))
	p_Gsrc = read_all_parameters(; kwargs...)
	@unpack Nt, dt = p_Gsrc

	if isnothing(τ_ind_range)
		τ_ind_range = [1:div(Nt, 2) for _ in xobs2_arr]
	end

	τ_ind_max_span = minimum(minimum.(τ_ind_range)):maximum(maximum.(τ_ind_range))

	Cτ_arr = [zeros(τ_inds) for τ_inds in τ_ind_range]

	@showprogress 1 "Cτ τ ind : " for τ_ind in τ_ind_max_span
		τ = (τ_ind-1) * dt
		xobs2′_arr = [shiftϕ(xobs2, -Ω_rot*τ)  for (rank,xobs2) in enumerate(xobs2_arr) if τ_ind in τ_ind_range[rank]]
		xobs2′inds_arr = [rank for rank in eachindex(xobs2_arr) if τ_ind in τ_ind_range[rank]]
		Ct_xobs2_arr = Ct(comm, xobs1, xobs2′_arr, los; kwargs...)[τ_ind, :]
		for (Ct_x2, x2ind) in zip(Ct_xobs2_arr, xobs2′inds_arr)
			Cτ_arr[x2ind][τ_ind] = Ct_x2
		end
	end

	return Cτ_arr
end

#######################################################################################################################################
# Allocate arrays
#######################################################################################################################################

function allocatearrays(::SoundSpeed, los::los_direction, obs_at_same_height)
	Gsrc = zeros(ComplexF64, nr, 0:1, srcindG(los)...)
	drGsrc = zeros(ComplexF64, nr, 0:0, srcindG(los)...)
	Gobs1 = zeros(ComplexF64, nr, 0:1, srcindG(los)...)
	drGobs1 = zeros(ComplexF64, nr, 0:0, srcindG(los)...)
	Gobs2, drGobs2 = Gobs1, drGobs1

	Gobs1_cache = Dict{Int, typeof(Gobs1)}()
	drGobs1_cache = Dict{Int, typeof(drGobs1)}()
	Gobs2_cache = obs_at_same_height ? Gobs1_cache : Dict{Int, typeof(Gobs2)}()
	drGobs2_cache = obs_at_same_height ? drGobs1_cache : Dict{Int, typeof(drGobs2)}()

	divGsrc = zeros(ComplexF64, nr, srcindG(los)...)
	divGobs = zeros(ComplexF64, nr, srcindG(los)...)

	# f_αjₒjₛ(r, rᵢ, rₛ) = -2ρc ∇⋅Gjₒ(r, rᵢ)_α ∇⋅Gjₛ(r, rₛ)_0
	fjₒjₛ_r₁_rsrc = zeros(ComplexF64, nr, srcindG(los)...)
	fjₒjₛ_r₂_rsrc = obs_at_same_height ? fjₒjₛ_r₁_rsrc : zero(fjₒjₛ_r₁_rsrc)

	# H_βαjₒjₛ(r; r₁, r₂, rₛ) = conj(f_αjₒjₛ(r, r₁, rₛ)) Gβ0jₛ(r₂, rₛ)
	Hjₒjₛω_r₁r₂ = zeros(ComplexF64, nr, obsindG(los)..., srcindG(los)...)
	Hjₒjₛω_r₂r₁ = obs_at_same_height ? Hjₒjₛω_r₁r₂ : zero(Hjₒjₛω_r₁r₂)

	tworealconjhωHjₒjₛω_r₁r₂ = zeros(nr, obsindG(los)..., srcindG(los)...)
	tworealconjhωconjHjₒjₛω_r₂r₁ = zero(tworealconjhωHjₒjₛω_r₁r₂)

	(; Gsrc, drGsrc, Gobs1, drGobs1, Gobs2, drGobs2,
		Gobs1_cache, drGobs1_cache, Gobs2_cache, drGobs2_cache,
		divGsrc, divGobs,
		fjₒjₛ_r₁_rsrc, fjₒjₛ_r₂_rsrc,
		Hjₒjₛω_r₁r₂, Hjₒjₛω_r₂r₁,
		tworealconjhωHjₒjₛω_r₁r₂, tworealconjhωconjHjₒjₛω_r₂r₁)
end

function allocatearrays(::Flow, los::los_direction, obs_at_same_height)
	Gsrc = zeros(ComplexF64, nr, 0:1, srcindG(los)...)
	drGsrc = zeros(ComplexF64, nr, 0:1, srcindG(los)...)
	Gobs1 = zeros(ComplexF64, nr, 0:1, srcindG(los)...)
	Gobs2 = Gobs1

	Gobs1_cache = Dict{Int, typeof(Gobs1)}()
	Gobs2_cache = Dict{Int, typeof(Gobs2)}()

	Gparts_r₁ = OffsetVector([zeros(ComplexF64, nr, 0:1, srcindG(los)...) for γ=0:1], 0:1)
	Gparts_r₂ = obs_at_same_height ? Gparts_r₁ : OffsetVector(
					[zeros(ComplexF64, nr, 0:1, srcindG(los)...) for γ=0:1], 0:1)

	# This is Gγℓjₒjₛω_α₁0(r, r₁, rₛ) as computed in the paper
	# Stored as Gγℓjₒjₛ_r₁[r,γ,αᵢ] = Gparts_r₁[γ][r, 0,αᵢ] + ζ(jₛ, jₒ, ℓ) Gparts_r₁[γ][:, 1,αᵢ]
	Gγℓjₒjₛ_r₁ = zeros(ComplexF64, nr, 0:1, srcindG(los)...)
	Gγℓjₒjₛ_r₂ = obs_at_same_height ? Gγℓjₒjₛ_r₁ : zero(Gγℓjₒjₛ_r₁)

	# This is given by Hγℓjₒjₛω_α₁α₂(r, r₁, r₂) = conj(Gγℓjₒjₛω_α₁0(r, r₁, rₛ)) * Gjₛω_α₂0(r₂, rₛ)
	# Stored as Hγℓjₒjₛ_r₁r₂[r,γ,α₁,α₂] = conj(Gγℓjₒjₛ_r₁[r,γ,α₁]) * Gα₂r_r₂[α₂]
	Hγℓjₒjₛ_r₁r₂ = zeros(ComplexF64, nr, 0:1, srcindG(los)..., obsindG(los)...)
	Hγℓjₒjₛ_r₂r₁ = obs_at_same_height ? Hγℓjₒjₛ_r₁r₂ : zero(Hγℓjₒjₛ_r₁r₂)

	twoimagconjhωHγℓjₒjₛ_r₁r₂ = zeros(nr, 0:1, srcindG(los)..., obsindG(los)...)
	twoimagconjhωconjHγℓjₒjₛ_r₂r₁ = zeros(nr, 0:1, srcindG(los)..., obsindG(los)...)

	# temporary array to save the γ=-1 component that may be used to compute the γ=1 one
	# the γ = 0 component is also stored here
	# temp = zeros(ComplexF64, nr)
	temp = StructArray{ComplexF64}((zeros(nr), zeros(nr)));

	# This is Gγℓjₒjₛ_r₁ for γ=1, used for the validation test
	G¹₁jj_r₁ = zeros(ComplexF64, nr, srcindG(los)...)
	G¹₁jj_r₂ = obs_at_same_height ? G¹₁jj_r₁ : zero(G¹₁jj_r₁)

	# Hjₒjₛαβ(r; r₁, r₂, rₛ) = conj(f_αjₒjₛ(r, r₁, rₛ)) Gβ0jₛ(r₂, rₛ)
	# We only use this for the validation case of ℑu⁺, so jₒ = jₛ = j
	H¹₁jj_r₁r₂ = zeros(ComplexF64, nr, srcindG(los)..., obsindG(los)...)
	H¹₁jj_r₂r₁ = obs_at_same_height ? H¹₁jj_r₁r₂ : zero(H¹₁jj_r₁r₂)

	(; Gsrc, Gobs1, Gobs2, drGsrc,
		Gobs1_cache, Gobs2_cache,
		temp,
		Gparts_r₁, Gparts_r₂,
		Gγℓjₒjₛ_r₁, Gγℓjₒjₛ_r₂,
		Hγℓjₒjₛ_r₁r₂, Hγℓjₒjₛ_r₂r₁,
		twoimagconjhωconjHγℓjₒjₛ_r₂r₁,
		twoimagconjhωHγℓjₒjₛ_r₁r₂,
		G¹₁jj_r₁, G¹₁jj_r₂,
		H¹₁jj_r₁r₂, H¹₁jj_r₂r₁)
end

#######################################################################################################################################
# First Born approximation for flows
#######################################################################################################################################

function δCωℓ_FB!(δC_r, ω_ind, ω, ℓ, Y12ℓ, Gsrc, G¹₁jj_r₂, G¹₁jj_r₁, r₁_ind, r₂_ind, ::los_radial)
	G_r₁_rsrc = Gsrc[r₁_ind, 0]
	G_r₂_rsrc = Gsrc[r₂_ind, 0]

	pre = ω^3 * Powspec(ω) * Y12ℓ

	@. δC_r[:, ω_ind] += pre *
		(conj(G_r₁_rsrc) * G¹₁jj_r₂ + G_r₂_rsrc * conj(G¹₁jj_r₁))
	return δC_r
end

function δCωℓ_FB!(δC_r, ω_ind, ω, ℓ, Y12ℓ, Gsrc, G¹₁jj_r₂, G¹₁jj_r₁, r₁_ind, r₂_ind, ::los_earth)
	for β in 0:1, γ in 0:1
		Gγ_r₁_rsrc = Gsrc[r₁_ind, γ, 0]
		Gβ_r₂_rsrc = Gsrc[r₂_ind, β, 0]

		gγ_r₁_rsrc = view(G¹₁jj_r₁, :, γ)
		gβ_r₂_rsrc = view(G¹₁jj_r₂, :, β)

		pre = ω^3 * Powspec(ω)* Ω(ℓ, 0) * √(1/6) * Y12ℓ[γ,β]

		@. δC_r[:, ω_ind] += pre *
			(conj(Gγ_r₁_rsrc) * gβ_r₂_rsrc + conj(gγ_r₁_rsrc) * Gβ_r₂_rsrc)
	end
	return δC_r
end

_integrate_δCrω(δC_r, ::los_radial) = -2im*integrate(r, (@. r^2 * ρ * δC_r))
_integrate_δCrω(δC_r, ::los_earth) = -4integrate(r, (@. r^2 * ρ * δC_r))

function δCω_uniform_rotation_firstborn_integrated_over_angle_partial(
	localtimer,
	ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D, xobs2::Point3D, los::los_direction,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, r_obs = nothing, Ω_rot = 20e2/Rsun)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src)
	p_Gobs1 = read_all_parameters(p_Gobs1, r_src = xobs1.r)
	p_Gobs2 = read_all_parameters(p_Gobs2, r_src = xobs2.r)

	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	Gfn_path_obs1, NGfn_files_obs1 =  p_Gobs1.path, p_Gobs1.num_procs
	Gfn_path_obs2, NGfn_files_obs2 = p_Gobs2.path, p_Gobs2.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	_, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2 =
			Gfn_fits_files.((Gfn_path_src, Gfn_path_obs1, Gfn_path_obs2),
			((ℓ_arr, 1:Nν_Gfn),), (ℓ_ωind_iter_on_proc,),
			(NGfn_files_src, NGfn_files_obs1, NGfn_files_obs2))

	δC_r = zeros(ComplexF64, nr, ν_ind_range)

	arrs = allocatearrays(Flow(), los, r₁_ind == r₂_ind)
	@unpack Gsrc, Gobs1, Gobs2, G¹₁jj_r₁, G¹₁jj_r₂ = arrs

	ℓ_range = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)
	Y12 = los_projected_biposh_spheroidal(computeY₁₀, xobs1, xobs2, los, ℓ_range)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		# Green function about source location
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, :, 1:2, srcindFITS(los), 1)

		# Green function about receiver location
		read_Gfn_file_at_index!(Gobs1, Gfn_fits_files_obs1,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs1, :, 1:2, srcindFITS(los), 1)

		radial_fn_uniform_rotation_firstborn!(G¹₁jj_r₁, Gsrc, Gobs1, ℓ, los)

		if r₁_ind != r₂_ind
			# Green function about receiver location
	    	read_Gfn_file_at_index!(Gobs2, Gfn_fits_files_obs2,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs2, :, 1:2, srcindFITS(los), 1)

			radial_fn_uniform_rotation_firstborn!(G¹₁jj_r₂, Gsrc, Gobs2, ℓ, los)
		end

		δCωℓ_FB!(δC_r, ω_ind, ω, ℓ, Y12[ℓ], Gsrc, G¹₁jj_r₂, G¹₁jj_r₁, r₁_ind, r₂_ind, los)
	end

	foreach(closeGfnfits, (Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2))

	δCω = Ω_rot * _integrate_δCrω(δC_r, los)#* integrate(r, (@. r^2 * ρ * δC_r))

	parent(δCω)
end

# 3D points
function δCω_uniform_rotation_firstborn_integrated_over_angle(comm,
	xobs1::Point3D, xobs2::Point3D, los::los_direction = los_radial(); kwargs...)

	r_src, r_obs = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(r_src = r_src)
	p_Gobs1 = read_all_parameters(r_src = xobs1.r)
	p_Gobs2 = read_all_parameters(r_src = xobs2.r)
	@unpack Nν_Gfn, ν_arr, ℓ_arr, ν_start_zeros, ν_end_zeros = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)

	Ω_rot = get(kwargs, :Ω_rot, 20e2/Rsun)

	Cω_in_range = pmapsum(comm,
		δCω_uniform_rotation_firstborn_integrated_over_angle_partial,
		iters, xobs1, xobs2, los, p_Gsrc, p_Gobs1, p_Gobs2, r_src, r_obs, Ω_rot)

	Cω_in_range === nothing && return nothing
	ν_ind_range = last(iters)
	δCω_flows_FB = pad_zeros_ν(Cω_in_range, ν_ind_range, Nν_Gfn, ν_start_zeros, ν_end_zeros)

	save_to_fits_and_return("δCω_flows_FB.fits", δCω_flows_FB)
end

# 2D points
@two_points_on_the_surface δCω_uniform_rotation_firstborn_integrated_over_angle

#######################################################################################################################################
# δC(ω) = C(ω) - C0(ω) for flows
########################################################################################################################################

# Linearized, without los, radial components, 3D points
function δCω_uniform_rotation_rotatedwaves_linearapprox(comm, xobs1, xobs2,
	los::los_radial = los_radial(); kwargs...)

	# We compute δC(xobs1, xobs2) = -iΩ ∑ℓ (2ℓ+1)/4π ∂ω (ω^2 P(ω) αℓω*(r₂)αℓω(r₁))∂Δϕ Pl(cos(Δϕ))

	r_src = get(kwargs, :r_src, r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src, c_scale = get(kwargs, :c_scale, 1))
	@load joinpath(Gfn_path_src,"parameters.jld2") dω

	∂ϕC = ∂ϕ₂Cω(comm, xobs1, xobs2, los; kwargs...)
	∂ϕC === nothing && return nothing

	Ω_rot = get(kwargs, :Ω_rot, 20e2/Rsun)
	∂ω∂ϕC = D(size(∂ϕC, 1))*∂ϕC ./ dω
	δCω_flows_rotated_linear = @. -im*Ω_rot*∂ω∂ϕC
	save_to_fits_and_return("δCω_flows_rotated_linear.fits", δCω_flows_rotated_linear)
end

# With or without los, 3D points
function δCω_uniform_rotation_rotatedwaves(comm, xobs1, xobs2, los::los_direction = los_radial(); kwargs...)
	dt = get(kwargs, :dt) do
		read_parameters("dt"; kwargs...)[1]
	end
	δCt = δCt_uniform_rotation_rotatedwaves(comm, xobs1, xobs2, los; kwargs...)
	δCt === nothing && return nothing
	δCω_flows_rotated = fft_t_to_ω(δCt, dt)
	save_to_fits_and_return("δCω_flows_rotated.fits", δCω_flows_rotated)
end

# With or without los, using precomputed time-domain cross-covariance
function δCω_uniform_rotation_rotatedwaves(δCt::AbstractArray; kwargs...)
	dt = get(kwargs, :dt) do
		read_parameters("dt"; kwargs...)[1]
	end
	δCω_flows_rotated = fft_t_to_ω(δCt, dt)
	return save_to_fits_and_return("δCω_flows_rotated.fits", δCω_flows_rotated)
end

#######################################################################################################################################
# δC(t) = C(t) - C0(t) for flows
########################################################################################################################################

# With or without los
function δCt_uniform_rotation_rotatedwaves(comm, xobs1, xobs2, los::los_direction = los_radial(); kwargs...)
	dν = get(kwargs, :dν) do
		read_parameters("dν"; kwargs...)[1]
	end

	C′_t = Cτ_rotating(comm, xobs1, xobs2, los; kwargs...)
	C0_t = get(kwargs, :Ct) do
		C0_ω = get(kwargs, :Cω) do
			Cω(comm, xobs1, xobs2, los; kwargs...)
		end
		C0_ω === nothing ? nothing : Ct(C0_ω; dν=dν, kwargs...)
	end
	C′_t === nothing && return nothing
	δCt_flows_rotated = parent(C′_t) .- parent(C0_t)
	save_to_fits_and_return("δCt_flows_rotated.fits", δCt_flows_rotated)
end

# With or without los, multiple points
function δCt_uniform_rotation_rotatedwaves(comm, xobs1, xobs2_arr::Vector,
	los::los_direction = los_radial(); kwargs...)

	C′_t = Cτ_rotating(comm, xobs1, xobs2_arr, los; kwargs...)
	C0_t = get(kwargs, :Ct) do
			[(ind,τ_ind) -> begin
					C = Ct(comm, xobs1, xobs2_arr, los; kwargs...)
					C === nothing && return nothing
					τ_ind = axes(C′_t[ind],1)
					C[τ_ind, ind]
				end
				for ind in eachindex(xobs2_arr)]
	end
	C′_t === nothing && return nothing
	return parent(C′_t) .- C0_t
end

# Linearized, with or without los
function δCt_uniform_rotation_rotatedwaves_linearapprox(comm, xobs1, xobs2,
	los::los_direction = los_radial(); kwargs...)

	C = ∂ϕ₂Ct(comm, xobs1, xobs2, los; kwargs...)
	C === nothing && return nothing
	Nt = read_parameters("Nt"; kwargs...)
	τ_ind_range = get(kwargs, :τ_ind_range, 1:Nt)
	t = fftfreq(Nt, dt*Nt)
	Ω_rot = get(kwargs, :Ω_rot, 20e2/Rsun)
	@views -Ω_rot .* t[τ_ind_range] .* C[τ_ind_range]
end

# Linearized, with or without los, multiple 3D points
function δCt_uniform_rotation_rotatedwaves_linearapprox(comm, xobs1::Point3D,
	xobs2_arr::Vector{<:Point3D}, los::los_direction = los_radial(); kwargs...)

	_, Nt = read_parameters("Nt"; kwargs...)

	t = fftfreq(Nt, dt*Nt)
	τ_ind_range = get(kwargs, :τ_ind_range, [1:Nt for _ in xobs2_arr])

	Ω_rot = get(kwargs, :Ω_rot, 20e2/Rsun)

	C = ∂ϕ₂Ct(comm, xobs1, xobs2_arr, los; kwargs...)
	C === nothing && return nothing
	δCt = -Ω_rot .* t .* C
	map(zip(τ_ind_range, eachindex(xobs2_arr))) do (τ_ind, x2ind)
		δCt[τ_ind, x2ind]
	end
end

# Linearized, with or without los, multiple 2D points
function δCt_uniform_rotation_rotatedwaves_linearapprox(comm, nobs1::Point2D,
	nobs2_arr::Vector{<:Point2D}, los::los_direction = los_radial(); kwargs...)

	C = ∂ϕ₂Ct(nobs1, nobs2_arr, los; kwargs...)
	C === nothing && return nothing
	δCt_uniform_rotation_rotatedwaves_linearapprox(C; kwargs...)
end

# Linearized, with or without los, using precomputed time-domain cross-covariance
function δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_arr; kwargs...)
	Nt, dt = read_parameters("Nt","dt"; kwargs...)

	τ_ind_range = get(kwargs, :τ_ind_range, 1:Nt)
	t = fftfreq(Nt, dt*Nt)

	Ω_rot = get(kwargs, :Ω_rot, 20e2/Rsun)

	@views @. -Ω_rot * t[τ_ind_range] * ∂ϕ₂Ct_arr[τ_ind_range]
end

#######################################################################################################################################
# First Born approximation for sound speed perturbation
#######################################################################################################################################

function δC_FB_soundspeed!(δC_rω, ω_ind, ω, Y12ℓ, H¹₁jj_r₂r₁, H¹₁jj_r₁r₂, ::los_radial)
	pre = ω^2*Powspec(ω)* Y12ℓ
	@. δC_rω[:, ω_ind] += pre * (conj(H¹₁jj_r₂r₁) + H¹₁jj_r₁r₂)
	return δC_rω
end

function δC_FB_soundspeed!(δC_rω, ω_ind, ω, Y12ℓ, Hjₒjₛω_r₂r₁, Hjₒjₛω_r₁r₂, ::los_earth)
	pre = ω^2 * Powspec(ω)
	for β in 0:1, α in 0:1
		@. δC_rω[:, ω_ind] += pre * Y12ℓ[α, β] * ( Hjₒjₛω_r₁r₂[:, α, β] + conj(Hjₒjₛω_r₂r₁[:, β, α]) )
	end
	return δC_rω
end

function δCω_isotropicδc_firstborn_integrated_over_angle_partial(localtimer,
	ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D, xobs2::Point3D, los::los_direction,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs1::Union{Nothing, ParamsGfn} = nothing,
	p_Gobs2::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, r_obs = nothing, c_scale = 1+1e-5)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = 1)
	p_Gobs1 = read_all_parameters(p_Gobs1, r_src = xobs1.r, c_scale = 1)
	p_Gobs2 = read_all_parameters(p_Gobs2, r_src = xobs2.r, c_scale = 1)

	Gfn_path_src, NGfn_files_src = p_Gsrc.path, p_Gsrc.num_procs
	Gfn_path_obs1, NGfn_files_obs1 =  p_Gobs1.path, p_Gobs1.num_procs
	Gfn_path_obs2, NGfn_files_obs2 = p_Gobs2.path, p_Gobs2.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	_, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	ϵ = c_scale-1

	Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2 =
		Gfn_fits_files.((Gfn_path_src, Gfn_path_obs1, Gfn_path_obs2),
		((ℓ_arr, 1:Nν_Gfn),), (ℓ_ωind_iter_on_proc,),
		(NGfn_files_src, NGfn_files_obs1, NGfn_files_obs2))

	δC_rω = zeros(ComplexF64, nr, ν_ind_range)

	arrs = allocatearrays(SoundSpeed(), los, r₁_ind == r₂_ind)
	@unpack Gsrc, drGsrc, divGsrc, Gobs1, drGobs1, Gobs2, divGobs = arrs
	@unpack Hjₒjₛω_r₁r₂, Hjₒjₛω_r₂r₁ = arrs
	@unpack fjₒjₛ_r₁_rsrc, fjₒjₛ_r₂_rsrc = arrs

	ℓ_range = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)
	Y12 = los_projected_biposh_spheroidal(computeY₀₀, xobs1, xobs2, los, ℓ_range)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		# Green function about the source location
		read_Gfn_file_at_index!(Gsrc, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, :, 1:2, srcindFITS(los), 1)

		Gγr_r₁_rsrc = αrcomp(Gsrc, r₁_ind, los)
		Gγr_r₂_rsrc = αrcomp(Gsrc, r₂_ind, los)

		# Derivative of Green function about the source location
		read_Gfn_file_at_index!(drGsrc, Gfn_fits_files_src,
		(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_src, :, 1:1, srcindFITS(los), 2)

		# Green function about the receiver location
		read_Gfn_file_at_index!(Gobs1, Gfn_fits_files_obs1,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs1, :, 1:2, srcindFITS(los), 1)

		# Derivative of Green function about the receiver location
		read_Gfn_file_at_index!(drGobs1, Gfn_fits_files_obs1,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs1, :, 1:1, srcindFITS(los), 2)

		radial_fn_isotropic_δc_firstborn!(fjₒjₛ_r₁_rsrc,
			Gsrc, drGsrc, divGsrc, Gobs1, drGobs1, divGobs, ℓ)

		Hjₒjₛω!(Hjₒjₛω_r₁r₂, fjₒjₛ_r₁_rsrc, Gγr_r₂_rsrc)

		if r₁_ind != r₂_ind
			# Green function about the receiver location
    		read_Gfn_file_at_index!(Gobs2, Gfn_fits_files_obs2,
    			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs2, :, 1:2, srcindFITS(los), 1)

    		# Derivative of Green function the about receiver location
    		read_Gfn_file_at_index!(drGobs2, Gfn_fits_files_obs2,
    			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_obs2, :, 1:1, srcindFITS(los), 2)

			radial_fn_isotropic_δc_firstborn!(fjₒjₛ_r₂_rsrc,
				Gsrc, drGsrc, divGsrc, Gobs2, drGobs2, divGobs, ℓ)

			Hjₒjₛω!(Hjₒjₛω_r₂r₁, fjₒjₛ_r₂_rsrc, Gγr_r₁_rsrc)
		end

		δC_FB_soundspeed!(δC_rω, ω_ind, ω, Y12[ℓ], Hjₒjₛω_r₂r₁, Hjₒjₛω_r₁r₂, los)
	end

	foreach(closeGfnfits, (Gfn_fits_files_src, Gfn_fits_files_obs1, Gfn_fits_files_obs2))

	C = parent(δC_rω)
	integrate(r, (@. r^2 * (ϵ*c) * C))
end

function _δCω_isotropicδc_firstborn(comm, xobs1, xobs2, args...; kwargs...)
	r_src, r_obs, c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc, p_Gobs1, p_Gobs2 = read_parameters_for_points(xobs1, xobs2; kwargs..., c_scale = 1)
	@unpack Nν_Gfn, ν_arr, ℓ_arr, ν_start_zeros, ν_end_zeros = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)

	δCω_in_range = pmapsum(comm,
		δCω_isotropicδc_firstborn_integrated_over_angle_partial,
		iters, xobs1, xobs2, args...,
		p_Gsrc, p_Gobs1, p_Gobs2, r_src, r_obs, c_scale)

	δCω_in_range === nothing && return nothing

	ν_ind_range = last(iters)
	pad_zeros_ν(δCω_in_range, ν_ind_range, Nν_Gfn, ν_start_zeros, ν_end_zeros)
end

# With our without los, 3D points
function δCω_isotropicδc_firstborn_integrated_over_angle(comm, xobs1::Point3D, xobs2::Point3D,
	los::los_direction = los_radial(); kwargs...)

	δCω_isotropicδc_FB = _δCω_isotropicδc_firstborn(comm, xobs1, xobs2, los; kwargs...)
	δCω_isotropicδc_FB === nothing && return nothing

	tag = los isa los_radial ? "" : "_los"
	filename = "δCω_isotropicδc_FB$tag.fits"
	save_to_fits_and_return(filename, δCω_isotropicδc_FB)
end

# 2D points
@two_points_on_the_surface δCω_isotropicδc_firstborn_integrated_over_angle

function δCt_isotropicδc_firstborn_integrated_over_angle(comm, xobs1, xobs2,
	los::los_direction = los_radial(); kwargs...)

	dν = read_parameters("dν"; kwargs...)[1]

	δCω = δCω_isotropicδc_firstborn_integrated_over_angle(comm, xobs1, xobs2, los; kwargs...)
	δCω === nothing && return nothing
	δCt_isotropicδc_FB = fft_ω_to_t(δCω, dν)

	tag = los isa los_radial ? "" : "_los"
	filename = "δCt_isotropicδc_FB$tag.fits"
	save_to_fits_and_return(filename, δCt_isotropicδc_FB)
end

#######################################################################################################################################
# δC(ω) = C(ω) - C0(ω) for sound speed perturbations
#######################################################################################################################################

function δCω_isotropicδc_C_minus_C0_partial(localtimer,
	ℓ_ωind_iter_on_proc::ProductSplit,
	xobs1::Point3D, xobs2::Point3D, los::los_earth,
	p_Gsrc::Union{Nothing, ParamsGfn} = nothing,
	p_G′src::Union{Nothing, ParamsGfn} = nothing,
	r_src = r_src_default, r_obs = nothing, c_scale = 1+1e-5)

	p_Gsrc = read_all_parameters(p_Gsrc, r_src = r_src, c_scale = 1)
	p_G′src = read_all_parameters(p_G′src, r_src = r_src, c_scale = c_scale)

	Gfn_path_src_c0, NGfn_files_c0 = p_Gsrc.path, p_Gsrc.num_procs
	Gfn_path_src_c′, NGfn_files_c′ =  p_G′src.path, p_G′src.num_procs
	@unpack ℓ_arr, ω_arr, Nν_Gfn = p_Gsrc

	ℓ_range, ν_ind_range = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	Gfn_fits_files_src_c0, Gfn_fits_files_src_c′ =
	Gfn_fits_files.((Gfn_path_src_c0, Gfn_path_src_c′), ((ℓ_arr, 1:Nν_Gfn),),
		(ℓ_ωind_iter_on_proc,), (NGfn_files_c0, NGfn_files_c′))

	Cω = zeros(ComplexF64, ν_ind_range)

	r₁_ind, r₂_ind = radial_grid_index.((xobs1, xobs2))

	G1_c′ = zeros(ComplexF64, 0:1)
	G1_c0 = zeros(ComplexF64, 0:1)
	if r₁_ind != r₂_ind
		G2_c′ = zeros(ComplexF64, 0:1)
		G2_c0 = zeros(ComplexF64, 0:1)
	else
		G2_c′ = G1_c′
		G2_c0 = G1_c0
	end

	ℓ_range = UnitRange(extremaelement(ℓ_ωind_iter_on_proc, dims = 1)...)
	Y12 = los_projected_biposh_spheroidal(computeY₀₀, xobs1, xobs2, los, ℓ_range)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		ω = ω_arr[ω_ind]

		# Green function about source location at observation point 1
		read_Gfn_file_at_index!(G1_c0, Gfn_fits_files_src_c0,
		(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_c0, r₁_ind, 1:2, 1, 1)

		read_Gfn_file_at_index!(G1_c′, Gfn_fits_files_src_c′,
		(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_c′, r₁_ind, 1:2, 1, 1)

		# Green function about source location at observation point 2
		if r₁_ind != r₂_ind
			read_Gfn_file_at_index!(G2_c0, Gfn_fits_files_src_c0,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_c0, r₂_ind, 1:2, 1, 1)

			read_Gfn_file_at_index!(G2_c′, Gfn_fits_files_src_c′,
			(ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), NGfn_files_c′, r₂_ind, 1:2, 1, 1)
		end

		Y12ℓ = Y12[ℓ]

		for β in 0:1, α in 0:1
			GG_c0 = conj(G1_c0[α]) * G2_c0[β]
			GG_c′ = conj(G1_c′[α]) * G2_c′[β]
			δGG = GG_c′ - GG_c0
			Cω[ω_ind] += ω^2 * Powspec(ω) * δGG * Y12ℓ[α, β]
		end
	end

	closeGfnfits(Gfn_fits_files_src_c′)
	closeGfnfits(Gfn_fits_files_src_c0)

	parent(Cω)
end

function _δCω_isotropicδc_C_minus_C0(comm, xobs1, xobs2,
	los::los_earth, args...; kwargs...)
	r_src, r_obs, c_scale = read_rsrc_robs_c_scale(kwargs)
	p_Gsrc = read_all_parameters(r_src = r_src, c_scale = 1)
	p_G′src = read_all_parameters(r_src = r_src, c_scale = c_scale)
	@unpack Nν_Gfn, ν_arr, ℓ_arr, ν_start_zeros, ν_end_zeros = p_Gsrc
	iters = ℓ_and_ν_range(ℓ_arr, ν_arr; kwargs...)

	Cω_in_range = pmapsum(comm,
		δCω_isotropicδc_C_minus_C0_partial,
		iters, xobs1, xobs2, los, args..., p_Gsrc, p_G′src,
		r_src, r_obs, c_scale)

	Cω_in_range === nothing && return nothing

	ν_ind_range = last(iters)
	pad_zeros_ν(Cω_in_range, ν_ind_range, Nν_Gfn, ν_start_zeros, ν_end_zeros)
end

# Without los, radial components, 3D points
function δCω_isotropicδc_C_minus_C0(comm, xobs1::Point3D, xobs2::Point3D,
	los::los_radial = los_radial(); kwargs...)

	c_scale = get(kwargs, :c_scale, 1+1e-5)

	C′ = Cω(comm, xobs1, xobs2, los; kwargs..., c_scale = c_scale)
	C0 = Cω(comm, xobs1, xobs2, los; kwargs..., c_scale = 1)

	C′ === nothing && return nothing

	δCω_isotropicδc_CmC0 = @. C′- C0
	save_to_fits_and_return("δCω_isotropicδc_CmC0.fits", δCω_isotropicδc_CmC0)
end

# With los, 3D points
function δCω_isotropicδc_C_minus_C0(comm, xobs1::Point3D, xobs2::Point3D, los::los_earth; kwargs...)
	δCω_isotropicδc_CmC0_los = _δCω_isotropicδc_C_minus_C0(comm, xobs1, xobs2, los; kwargs...)
	δCω_isotropicδc_CmC0_los === nothing && return nothing
	save_to_fits_and_return("δCω_isotropicδc_CmC0_los.fits", δCω_isotropicδc_CmC0_los)
end

# 2D points
@two_points_on_the_surface δCω_isotropicδc_C_minus_C0

########################################################################################
# δC(t) = C(t) - C0(t) for sound speed perturbations
########################################################################################

# With or without los, 3D points
function δCt_isotropicδc_C_minus_C0(comm, xobs1, xobs2,
	los::los_direction = los_radial(); kwargs...)

	dν = read_parameters("dν"; kwargs...)[1]

	δCω = δCω_isotropicδc_C_minus_C0(comm, xobs1, xobs2, los; kwargs...)
	δCω === nothing && return nothing
	δCt_isotropicδc_CmC0 = fft_ω_to_t(δCω, dν)

	tag = los isa los_radial ? "" : "_los"
	filename = "δCt_isotropicδc_CmC0$tag.fits"
	save_to_fits_and_return(filename, δCt_isotropicδc_CmC0)
end

# With or without los, precomputed arrays
function δCt_isotropicδc_C_minus_C0(δCω::AbstractArray; kwargs...)
	dν = read_parameters("dν"; kwargs...)[1]
	δCt_isotropicδc_CmC0 = fft_ω_to_t(δCω, dν)
	save_to_fits_and_return("δCt_isotropicδc_CmC0.fits", δCt_isotropicδc_CmC0)
end

########################################################################################################################
# Window function
########################################################################################################################

function bounce_filter(Δϕ, n)
	nparams = 5
	coeffs = Dict()
	for i in [1, 2, 4]
		coeffs[i] = Dict("high"=>zeros(nparams),"low"=>zeros(nparams))
	end

	coeffs[1]["high"] = [2509.132334896018, 12792.508296270391,-13946.527195127102, 8153.75242742649,-1825.7567469552703]
	coeffs[1]["low"] = [40.821191938380714, 11410.21390421857,-11116.305124138207, 5254.244817703224,-895.0009393800744]

	coeffs[2]["high"] = [4083.6946001848364, 14924.442447995087,-13880.238239469609, 7562.499279468063,-1622.5318939228978]
	coeffs[2]["low"] = [2609.4406668522433, 10536.81683213881,-7023.811081076518, 2586.7238222832298,-348.9245124332354]

	coeffs[4]["high"] = [6523.103468645263, 16081.024611219753,-7085.7174198723405, 973.4990690666436, 236.95568587146957]
	coeffs[4]["low"] = [5150.314633252216, 15040.045600508669,-8813.047362534506, 3878.5398150601663,-870.3633232120256]

	τ_low,τ_high = 0., 0.
	for (i, c) in enumerate(coeffs[n]["high"])
		τ_high += c*Δϕ^(i-1)
	end

	for (i, c) in enumerate(coeffs[n]["low"])
		τ_low += c*Δϕ^(i-1)
	end

	return τ_low,τ_high
end

function gaussian_fit(x, y)
	# use the fact that if y = Gaussian(x), log(y) = quadratic(x)
	# quadratic(x) = ax² + bx + c
	# Gaussian(x) = A*exp(-(x-x0)²/2σ²)
	# the parameters are σ=√(-1/2a), x0 = -b/2a, A = exp(c-b^2/4a)
	c, b, a = polyfit(x, log.(y), 2).a
	A = exp(c-b^2/4a)
	x0 = -b/2a
	σ = √(-1/2a)
	return A, x0, σ
end

function time_window_bounce_filter(xobs1, xobs2, dt, bounce_no = 1)
	time_window_bounce_filter(acos(cosχ(xobs1, xobs2)), dt, bounce_no)
end

function time_window_bounce_filter(Δϕ::Real, dt, bounce_no = 1)
	τ_low,τ_high = bounce_filter(Δϕ, bounce_no)
	τ_low_ind = floor(Int64,τ_low/dt);
	τ_high_ind = ceil(Int64,τ_high/dt)
	return τ_low_ind,τ_high_ind
end

function time_window_indices_by_fitting_bounce_peak(C_t::AbstractVector{Float64},
	xobs1, xobs2; dt, bounce_no = 1, kwargs...)

	τ_low_ind,τ_high_ind = time_window_bounce_filter(xobs1, xobs2, dt, bounce_no)
	time_window_indices_by_fitting_bounce_peak(C_t,τ_low_ind,τ_high_ind; dt = dt, kwargs...)
end

hilbertenvelope(C_t, Nt = size(C_t, 1)) = abs.(hilbert(C_t[1:div(Nt, 2)]))

function time_window_indices_by_fitting_bounce_peak(C_t::AbstractVector{Float64},
	τ_low_ind::Int64, τ_high_ind::Int64;
	dt, Nt = size(C_t, 1), kwargs...)

	env = hilbertenvelope(C_t)
	peak_center = argmax(env[τ_low_ind:τ_high_ind]) + τ_low_ind - 1
	points_around_max = env[peak_center-2:peak_center+2]
	_, t0,σt = gaussian_fit(peak_center-2:peak_center+2, points_around_max)

	max(1, floor(Int64, t0 - 2σt)):min(ceil(Int64, t0 + 2σt), Nt)
end

function time_window_indices_by_fitting_bounce_peak(C_t::AbstractMatrix{Float64},
	xobs1, xobs2_arr::Vector, args...; kwargs...)

	t_inds_range = Vector{UnitRange}(undef, size(C_t, 2))
	for (x2ind, xobs2) in enumerate(xobs2_arr)
		t_inds_range[x2ind] = time_window_indices_by_fitting_bounce_peak(C_t[:, x2ind],
								xobs1, xobs2; kwargs...)
	end
	return t_inds_range
end

function time_window(a::Vector, τ_ind_range)
	b = zero(a)
	b[τ_ind_range] .= 1
	return b
end

function time_window(a::Matrix, τ_ind_range)
	b = zero(a)
	for (idx, τ_inds) in enumerate(τ_ind_range)
		b[τ_inds, idx] .= 1
	end
	return b
end

function ht(::TravelTimes, Cω_x1x2::AbstractArray, xobs1, xobs2; bounce_no = 1, kwargs...)
	Cω_x1x2 === nothing && return nothing
	p_Gsrc = read_all_parameters(; kwargs...)
	@unpack ν_full, Nt, dt, dν = p_Gsrc

	ω_full = 2π.*ν_full

	C_t = fft_ω_to_t(Cω_x1x2, dν)
	∂tCt_ω = @. Cω_x1x2*im*ω_full
	∂tCt = fft_ω_to_t(∂tCt_ω, dν)

	τ_ind_range = get(kwargs, :τ_ind_range, nothing)
	if isnothing(τ_ind_range)
		τ_ind_range = time_window_indices_by_fitting_bounce_peak(C_t, xobs1, xobs2;
					dt = dt, Nt = Nt, bounce_no = bounce_no)
	end

	f_t = time_window(∂tCt, τ_ind_range)

	(@. -f_t * ∂tCt) ./ sum((@. f_t * ∂tCt^2 * dt), dims = 1)
end

function ht(::Amplitudes, Cω_x1x2::AbstractArray, xobs1, xobs2; bounce_no = 1, kwargs...)
	Cω_x1x2 === nothing && return nothing
	Nt, dt, dν = read_parameters("Nt","dt","dν"; kwargs...)

	C_t = fft_ω_to_t(Cω_x1x2, dν)

	τ_ind_range = get(kwargs, :τ_ind_range, nothing)
	if isnothing(τ_ind_range)
		τ_ind_range = time_window_indices_by_fitting_bounce_peak(C_t, xobs1, xobs2;
					dt = dt, Nt = Nt, bounce_no = bounce_no)
	end

	f_t = time_window(C_t, τ_ind_range)

	(@. f_t * C_t) ./ sum((@. f_t*C_t^2 * dt), dims = 1)
end

function ht(comm, m::SeismicMeasurement, xobs1, xobs2, los::los_direction = los_radial(); kwargs...)
	Cω_x1x2 = Cω(comm, xobs1, xobs2, los; kwargs...)
	Cω_x1x2 === nothing && return nothing
	ht(m, Cω_x1x2, xobs1, xobs2; kwargs...)
end

function hω(args...; kwargs...)
	h_t = ht(args...; kwargs...)
	h_t === nothing && return nothing
	dt, ν_start_zeros, ν_arr = read_parameters("dt", "ν_start_zeros", "ν_arr"; kwargs...)
	h_ω = fft_t_to_ω(h_t, dt)
	ν_ind_range = get(kwargs, :ν_ind_range, axes(ν_arr, 1)) .+ ν_start_zeros
	h_ω[ν_ind_range, ..]
end
