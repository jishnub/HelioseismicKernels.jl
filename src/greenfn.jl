################################################################################
# Utility functions
################################################################################

# Indexing

srcindFITS(::los_radial) = 1
srcindFITS(::los_earth) = 1:2

srcindG(::los_radial) = ()
srcindG(::los_earth) = (0:1,)

obsindFITS(::los_radial) = 1
obsindFITS(::los_earth) = 1:2

obsindG(::los_radial) = ()
obsindG(::los_earth) = (0:1,)

sizeindG(::los_radial) = ()
sizeindG(::los_earth) = (2,)

αrcomp(G::AbstractArray{ComplexF64, 2}, r_ind, α) = G[α, r_ind]
αrcomp(G::AbstractArray{ComplexF64, 3}, r_ind, α) = G[α, 0, r_ind]
αrcomp(G::AbstractArray{ComplexF64, 2}, r_ind, ::los_radial) = αrcomp(G, r_ind, 0)
αrcomp(G::AbstractArray{ComplexF64, 3}, r_ind, ::los_radial) = αrcomp(G, r_ind, 0)
function αrcomp(G::AbstractArray{ComplexF64, 2}, r_ind, los::los_earth)
	αrcomp(G, r_ind, Base.IdentityUnitRange(0:1))
end
function αrcomp(G::AbstractArray{ComplexF64, 3}, r_ind, los::los_earth)
	αrcomp(G, r_ind, Base.IdentityUnitRange(0:1))
end

##################################################################################
# Compute the processor range for a particular range of modes
##################################################################################

function ℓ′ω_range_from_ℓω_range(ℓ_ωind_iter_on_proc::ProductSplit, s_max, ℓ_arr)
	ω_ind_min, ω_ind_max = extremaelement(ℓ_ωind_iter_on_proc, dims = 2)
	ℓ′min_ωmin = minimum(minimum(intersect(ℓ_arr, max(ℓ-s_max, 0):ℓ+s_max))
		for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc if ω_ind==ω_ind_min)
	ℓ′max_ωmax = maximum(maximum(intersect(ℓ_arr, max(ℓ-s_max, 0):ℓ+s_max))
		for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc if ω_ind==ω_ind_max)
	return (ℓ′min_ωmin, ω_ind_min),(ℓ′max_ωmax, ω_ind_max)
end

function Gfn_processor_range((ℓ_arr, ω_ind_arr), ℓ_ωind_iter_on_proc::ProductSplit, s_max, num_procs)
	modes = ℓ′ω_range_from_ℓω_range(ℓ_ωind_iter_on_proc, s_max, ℓ_arr)
	proc_id_min = ParallelUtilities.whichproc((ℓ_arr, ω_ind_arr), first(modes), num_procs)
	proc_id_max = ParallelUtilities.whichproc((ℓ_arr, ω_ind_arr), last(modes), num_procs)
	return proc_id_min:proc_id_max
end

###################################################################################
## Read Green function slices (real and imaginary parts) and assign to arrays
###################################################################################

function read_Gfn_file_at_index!(G::AbstractArray{<:Complex},
	G_fits_dict::Dict, (ℓ_arr, ω_ind_arr)::Tuple{Vararg{AbstractUnitRange, 2}},
	ℓω::NTuple{2, Int}, num_procs::Integer, I...)

	proc_id_mode, ℓω_index_in_file = ParallelUtilities.whichproc_localindex((ℓ_arr, ω_ind_arr), ℓω, num_procs)
	G_file = G_fits_dict[proc_id_mode].hdu
	read_Gfn_file_at_index!(G, G_file, I..., ℓω_index_in_file)
end

_maybetoreim(I1::Colon) = I1
_maybetoreim(I1::Union{AbstractUnitRange,Integer}) = 2first(I1)-1:2last(I1)
function _reimindices(I)
	ind1 = _maybetoreim(I[1])
	Itrailing = I[2:end]
	return ind1, Itrailing
end

function read_Gfn_file_at_index!(G::AbstractArray{<:Complex,0}, G_hdu::ImageHDU, I...)
	G1D = reshape(G, 1)
	read!(G_hdu, reinterpret_as_float(G1D), :, I...)
	return G
end
function read_Gfn_file_at_index!(G, G_hdu::ImageHDU, I...)
	read!(G_hdu, reinterpret_as_float(G), :, I...)
	return G
end

function read_Gfn_file_at_index(G_file::Union{FITS, ImageHDU},
	(ℓ_arr, ω_ind_arr)::NTuple{2, AbstractUnitRange},
	ℓω::NTuple{2, Int}, num_procs::Integer, I...)

	_, ℓω_index_in_file = ParallelUtilities.whichproc_localindex((ℓ_arr, ω_ind_arr), ℓω, num_procs)
	read_Gfn_file_at_index(G_file, I..., ℓω_index_in_file)
end

read_Gfn_file_at_index(G_hdu::ImageHDU, I...) =
	dropdims(reinterpret_as_complex(read(G_hdu, :, I...)), dims = 1)

function Gfn_fits_files(path::String, proc_id_range::AbstractUnitRange)
	function f(procid, path)
		filepath = joinpath(path, @sprintf "Gfn_proc_%03d.fits" procid)
		f_FITS = FITS(filepath,"r") :: FITS
		f_HDU = f_FITS[1] :: ImageHDU{Float64,6}
		(fits = f_FITS, hdu = f_HDU)
	end
	Dict(procid => f(procid, path) for procid in proc_id_range)
end

function Gfn_fits_files(path::String, (ℓ_arr, ω_ind_arr), ℓ_ωind_iter_on_proc::ProductSplit, num_procs::Integer)
	proc_range = ParallelUtilities.procrange_recast((ℓ_arr, ω_ind_arr), ℓ_ωind_iter_on_proc, num_procs)
	Gfn_fits_files(path, proc_range)
end

function Gfn_fits_files(path::String, (ℓ_arr, ω_ind_arr), ℓ_ωind_iter_on_proc::ProductSplit, s_max::Integer, num_procs::Integer)
	proc_range = Gfn_processor_range((ℓ_arr, ω_ind_arr), ℓ_ωind_iter_on_proc, s_max, num_procs)
	Gfn_fits_files(path, proc_range)
end

closeGfnfits(d::Dict) = map(x -> close(x.fits), values(d))

# Directory to save output to
function Gfn_path_from_source_radius(r_src::Real; c_scale = 1)
	dir = "Greenfn_src$((r_src/Rsun > 0.99 ?
			(@sprintf "%dkm" (r_src-Rsun)/1e5) :
			(@sprintf "%.2fRsun" r_src/Rsun)))"
	if c_scale != 1
		dir *= "_c_scale_$(@sprintf "%g" c_scale)"
	end
	dir *= "_flipped"
	joinpath(SCRATCH[], dir)
end
Gfn_path_from_source_radius(x::Point3D; kwargs...) = Gfn_path_from_source_radius(x.r; kwargs...)

#####################################################################################
# Solve for components
#####################################################################################

function load_solar_model()

	modelS_meta = readdlm("$(@__DIR__)/ModelS.meta", comments = true, comment_char='#')::Matrix{Float64};
	Msun, Rsun = modelS_meta;

	modelS = readdlm("$(@__DIR__)/ModelS", comments = true, comment_char='#')::Matrix{Float64};
	modelS_detailed = readdlm("$(@__DIR__)/ModelS.detailed", comments = true, comment_char='#')::Matrix{Float64};

	HMIfreq = readdlm("$(@__DIR__)/m181q.1216")::Matrix{Float64};
	ℓ_HMI::Vector{Int}, ν_HMI::Vector{Float64}, γ_HMI::Vector{Float64} = HMIfreq[:, 1], HMIfreq[:, 3], HMIfreq[:, 5];

	# Fit modes above ℓ=11 and 2mHz<ν<4mHz
	mode_filter = @. (ℓ_HMI > 11) & (ν_HMI > 1.5e3) & (ν_HMI < 4.5e3);

	ν_HMI = ν_HMI[mode_filter];
	ℓ_HMI = ℓ_HMI[mode_filter];
	γ_HMI = γ_HMI[mode_filter];

	# Fit γ(ω) in Hz, the HMI frequencies are in μHz
	γ_damping = polyfit(ν_HMI.*(2π*1e-6), γ_HMI.*(2π*1e-6), 3);

	# invert the grid to go from inside to outside,
	# and leave out the center to avoid singularities
	# Start from r = 0.2Rsun to compare with Mandal. et al 2017

	#r_start_ind_mandal = searchsortedfirst(modelS[:, 1], 0.2, rev = true)
	r_start_ind_skipzero::Int = searchsortedfirst(modelS[:, 1], 2e-2, rev = true)
	#r_start_ind_somewhere_inside = searchsortedfirst(modelS[:, 1], 0.5, rev = true)

	#set r_start_ind to r_start_ind_mandal to compare with Mandal. et al 2017
	r_start_ind = r_start_ind_skipzero

	r::Vector{Float64} = modelS[r_start_ind:-1:1, 1]*Rsun;

	nr = length(r);

	dr = D(nr; stencil_gridpts = Dict(7=>3, 43=>5))*r;
	ddr = sparse(dbydr(dr));
	c = modelS[r_start_ind:-1:1, 2]::Vector{Float64};
	ρ = modelS[r_start_ind:-1:1, 3]::Vector{Float64};

	G = 6.67428e-8 # cgs units
	m = @. Msun*exp(modelS_detailed[r_start_ind:-1:1, 2]::Vector{Float64})

	g = @. G*m/r^2

	N2 = @. g * modelS_detailed[r_start_ind:-1:1, 15]::Vector{Float64} / r

	return Rsun, nr, r, dr, ddr, c,ρ, g, N2,γ_damping
end

const Rsun, nr, r, dr, ddr, c, ρ, g, N2, γ_damping = load_solar_model()

radial_grid_index(x::Point3D) = radial_grid_index(x.r)
radial_grid_index(r_pt::Real) = radial_grid_index(r, r_pt)
radial_grid_index(r, r_pt::Real) = searchsortedfirst(r, r_pt)
radial_grid_closest(x::Point3D) = radial_grid_closest(x.r)
radial_grid_closest(r_pt::Real) = radial_grid_closest(r, r_pt)
radial_grid_closest(r, r_pt::Real) = r[radial_grid_index(r, r_pt)]

const r_src_default = radial_grid_closest(r, Rsun - 75e5)
const r_obs_default = radial_grid_closest(r, Rsun + 150e5)

function read_rsrc_robs_c_scale(kwargs)
	r_src = get(kwargs, :r_src, r_src_default)
	r_obs = get(kwargs, :r_obs, r_obs_default)
	c_scale = get(kwargs, :c_scale, 1)
	r_src, r_obs, c_scale
end

# Source components in Hansen VSH basis
function σsrc_grid(r_src = r_src_default)
	r_src_ind = radial_grid_index(r, r_src)
	r_src_on_grid = r[r_src_ind];
	max(r_src_on_grid - r[max(1, r_src_ind-2)], r[min(nr, r_src_ind+2)] - r_src_on_grid)
end

delta_fn_gaussian_approx(r_src, σsrc) = @. exp(-(r - r_src)^2 / 2σsrc^2) / √(2π*σsrc^2) /r^2;

function source!(Sh, ω, ℓ, delta_fn)
    @views @. Sh[nr+1:end] = -√(ℓ*(ℓ+1))/ω^2 * delta_fn[2:nr-1] /(r[2:nr-1]*ρ[2:nr-1])
    return Sh;
end

function ℒr!(L14, L22, L33, L41, c′; stencil_gridpts = Dict(6=>3, 42=>5), kw...)
    # Left edge ghost to use point β(r_in) in computing derivatives
    # Dirichlet condition β(r_out)=0 on the right edge
    dbydr!(L14, (@view dr[2:end-1]), stencil_gridpts = stencil_gridpts,
                left_edge_ghost = true, left_edge_npts = 3,
                right_edge_ghost = false, right_edge_npts = 3,
                right_Dirichlet = true) # (nr-2 x nr-1)

    dinds = diagind(L14, 1)
    @views @. L14[dinds] += g[2:nr-1]/c′[2:nr-1]^2

    # Boundary condition on α(r_out)
    bc_val = 2/r[end] - g[end]/c′[end]^2
    derivStencil!(L22, 1, 2, 0, gridspacing = dr[end])
    L22[end] += bc_val

    # Boundary condition on β(r_in)
    bc_val = g[1]/c′[1]^2
    derivStencil!(L33, 1, 0, 2, gridspacing = dr[1])
    L33[1] += bc_val

    # Right edge ghost to use point α(r_out) in computing derivatives
    # Dirichlet condition α(r_in)=0 on the left edge
    dbydr!(L41, (@view dr[2:end-1]), stencil_gridpts = stencil_gridpts,
                left_edge_ghost = false, left_edge_npts = 3,
                right_edge_ghost = true, right_edge_npts = 3,
                left_Dirichlet = true) # (nr-2 x nr-1)

    dinds = diagind(L41, 0)
    @views @. L41[dinds] += (2/r - g/c′^2)[2:end-1]

	return L14, L22, L33, L41
end

Ω(ℓ, N) = √((ℓ+N)*(ℓ-N+1)/2)
ζjₒjₛs(j₁, j₂, ℓ) = (Ω(j₁, 0)^2 + Ω(j₂, 0)^2 - Ω(ℓ, 0)^2)/(Ω(j₁, 0)*Ω(j₂, 0))
Njₒjₛs(jₒ, jₛ, s) = √((2jₒ+1)*(2jₛ+1)/(4π*(2s+1)))

function copyto_fixedterms!(M, L14, L22, L33, L41, invρc′², ρN2′)
	# top left
	dinds = diagind(M)[1:nr-2]
	for (ind1, ind2) in zip(dinds, 2:nr-1)
    	M[ind1] = ρN2′[ind2]
	end

	# top right, ∂ᵣβ, (nr-2 x nr-1)
    M[1:nr-2, end-(nr-2):end] = L14

	# Boundary condition on α(r_out) (∂ᵣ + 2/r - g/c^2)α(r_out)
    M[nr-1, nr-3:nr-1] = L22

	# Boundary condition on β(r_in) (∂ᵣ + g/c^2)β(r_in)
    M[nr, nr:nr+2] = L33

	# bottom left ∂ᵣα, (nr-2 x nr-1)
    M[nr+1:end, 1:nr-1] = L41

	# bottom right
    dinds = diagind(M)[nr+1:end]
	for (ind1, ind2) in zip(dinds, 2:nr-1)
    	M[ind1] = invρc′²[ind2]
	end
	return M
end

function ℒωℓr!(M_ωℓ, ℓωind, ω, ℓ, L14, L22, L33, L41, c′, invρc′², ρN2′, invρr²; kw...)

	if ℓωind == 1
		ℒr!(L14, L22, L33, L41, c′; kw...)
	end
	ωC = ω - im * γ_damping(ω)

    dinds1 = diagind(M_ωℓ)[1:nr-2]
	M_ωℓ[dinds1] .= 0
    dinds2 = diagind(M_ωℓ)[nr+1:end]
	M_ωℓ[dinds2] .= 0

	copyto_fixedterms!(M_ωℓ, L14, L22, L33, L41, invρc′², ρN2′)
	negω² = -ωC^2
	ℓℓp1overnegω² = ℓ*(ℓ+1) / negω²
	for (ind1, ind2) in zip(dinds1, 2:nr-1)
		M_ωℓ[ind1] += negω² * ρ[ind2]
	end
	for (ind1, ind2) in zip(dinds2, 2:nr-1)
		M_ωℓ[ind1] += ℓℓp1overnegω² * invρr²[ind2]
	end

	return M_ωℓ
end

# Functions to write the frequency grid and other parameters to the data directory

function frequency_grid(Gfn_save_directory; kwargs...)
	ν_low = get(kwargs, :ν_low, 2.0e-3)
	ν_high = get(kwargs, :ν_high, 4.5e-3)
	num_ν=get(kwargs, :num_ν, 1250)
	@assert(num_ν>1,"Need at least two points in frequency to construct the grid")
	ν_Nyquist = get(kwargs, :ν_Nyquist, 16e-3)

	ℓ_arr = get(kwargs, :ℓ_arr, 1:1)

	dν = (ν_high - ν_low)/(num_ν-1); dω = 2π*dν

	# choose values on a grid
	ν_low_index = Int64(floor(ν_low/dν)); ν_low = ν_low_index*dν
	ν_high_index = num_ν + ν_low_index - 1; ν_high = ν_high_index*dν;
	Nν_Gfn = num_ν
	ν_Nyquist_index = Int64(ceil(ν_Nyquist/dν)); ν_Nyquist = ν_Nyquist_index*dν

	Nν = ν_Nyquist_index + 1; Nt = 2*(Nν-1)
	ν_full = (0:ν_Nyquist_index).*dν;
	ν_arr = (ν_low_index:ν_high_index).*dν ;
	T = 1/dν; dt = T/Nt;
	ν_start_zeros = ν_low_index # index starts from zero
	ν_end_zeros = ν_Nyquist_index - ν_high_index

	ω_arr = 2π .* ν_arr;

	if !isdir(Gfn_save_directory)
		mkdir(Gfn_save_directory)
	end

	@save(joinpath(Gfn_save_directory,"parameters.jld2"),
		ν_arr, ω_arr, ν_full, dν, dω, ℓ_arr,
		ν_start_zeros, ν_end_zeros, Nν, Nt, dt, T, Nν_Gfn, ν_Nyquist)

	ℓ_arr, ω_arr
end

function append_parameters(Gfn_save_directory; kwargs...)
	paramfile = joinpath(Gfn_save_directory,"parameters.jld2")
	params = jldopen(paramfile,"a+")
	for (k, v) in Dict(kwargs)
		params[string(k)] = v
	end
	close(params)
end

function update_parameters(Gfn_save_directory; kwargs...)
	paramfile = joinpath(Gfn_save_directory,"parameters.jld2")
	params = load(paramfile)
	for (k, v) in Dict(kwargs)
		params[string(k)] = v
	end
	save(paramfile, params)
end

struct ParamsGfn
	path::String
	Nν_Gfn::Int
	ν_Nyquist::Float64
	ℓ_arr::UnitRange{Int}
	dν::Float64
	dω::Float64
	Nν::Int
	ν_full::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
	ν_arr::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
	T::Float64
	dt::Float64
	Nt::Int
	ν_start_zeros::Int
	ν_end_zeros::Int
	ω_arr::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
	num_procs::Int
end

function read_all_parameters(; kwargs...)
	r_src = get(kwargs, :r_src, r_src_default)
	c_scale = get(kwargs, :c_scale, 1)
	Gfn_path_src = Gfn_path_from_source_radius(r_src, c_scale = c_scale)
	# load all parameters from the file at the path
	params_all = load(joinpath(Gfn_path_src,"parameters.jld2"))
	# pack the parameters including the path into a struct
	ParamsGfn(Gfn_path_src,
		[params_all[String(k)] for k in fieldnames(ParamsGfn) if k != :path]...)
end

read_all_parameters(p::ParamsGfn; kwargs...) = p
function read_all_parameters(::Nothing; kwargs...)
	read_all_parameters(; kwargs...) :: ParamsGfn
end

function read_parameters(params...; kwargs...)
	isempty(params) && return ()
	params_all = read_all_parameters(; kwargs...)
	Tuple(getproperty(params_all, Symbol(p)) for p in params)
end

get_numprocs(path) = load(joinpath(path, "parameters.jld2"), "num_procs")

# Functions to compute Green function components in the helicity basis
# We solve in the Hansen VSH basis first and change over to the helicity basis

function solve_for_components!(M, S, α, β)

	H = M\S # solve the equation

	@views @. α[2:nr] = H[1:nr-1]
	@views @. β[1:nr-1] = H[nr:end]
    return α, β
end

function greenfn_components_onemode!(ℓωind, ω, ℓ, αrℓω, βrℓω, αhℓω, βhℓω, M_ωℓ,
	L14, L22, L33, L41, Sr, Sh, delta_fn, c′, invρc′², ρN2′, invρr²; kwargs...)

	tangential_source = get(kwargs, :tangential_source, true)

	source!(Sh, ω, ℓ, delta_fn)

	ℒωℓr!(M_ωℓ, ℓωind, ω, ℓ, L14, L22, L33, L41, c′, invρc′², ρN2′, invρr²; kwargs...);

    M_sparse = sparse(M_ωℓ)
	M_lu = lu(M_sparse)

	# radial source
	solve_for_components!(M_lu, Sr, αrℓω, βrℓω)

	if tangential_source
		solve_for_components!(M_lu, Sh, αhℓω, βhℓω)
	end
    return nothing
end

function greenfn_components_somemodes_serial_oneproc(ℓ_ωind_proc::ProductSplit,
	r_src, c_scale,
	ω_arr, Gfn_save_directory,
	tracker::Union{Nothing, RemoteChannel} = nothing;
	tangential_source::Bool = true,
    rank = ParallelUtilities.workerrank(ℓ_ωind_proc))

	save_path = joinpath(Gfn_save_directory, @sprintf "Gfn_proc_%03d.fits" rank)

	FITS(save_path, "w") do file

		# save real and imaginary parts separately
		β = 0:1
		γ = tangential_source ? (0:1) : (0:0)
		G = zeros(ComplexF64, nr, β, γ, 0:1, length(ℓ_ωind_proc))

		r_src_on_grid = radial_grid_closest(r, r_src)
		σsrc = σsrc_grid(r_src_on_grid)
		delta_fn = delta_fn_gaussian_approx(r_src_on_grid, σsrc)
        Sr, Sh = zeros(2nr-2), zeros(2nr-2);
		@views @. Sr[1:nr-2] = delta_fn[2:nr-1]

		αr, βr = zeros(ComplexF64, nr), zeros(ComplexF64, nr);
		αh, βh = zeros(ComplexF64, nr), zeros(ComplexF64, nr);

        # temporary arrys used to compute the derivative operator
        M_ωℓ = zeros(ComplexF64, 2(nr-1), 2(nr-1));
        L14 = zeros(nr-2, nr-1);
        L41 = zero(L14);
        L22 = zeros(3);
        L33 = zeros(3);

        stencil_gridpts = Dict(6=>3, 42=>5);

		if c_scale != 1
			c′= c .* c_scale
			N2′= @. N2 + g^2/c^2 * (1-1/c_scale^2)
		else
			c′ = c;
			N2′ = N2;
		end
		invρc′² = @. 1 / (ρ * c′^2);
		ρN2′ = ρ .* N2′;
		invρr² = @. 1 / (ρ * r^2)

		for (ℓωind, (ℓ, ω_ind)) in enumerate(ℓ_ωind_proc)

			ω = ω_arr[ω_ind]

			greenfn_components_onemode!(ℓωind, ω, ℓ, αr,
				βr, αh, βh, M_ωℓ, L14, L22, L33, L41, Sr, Sh, delta_fn,
				c′, invρc′², ρN2′, invρr²,
                stencil_gridpts = stencil_gridpts,
				r_src = r_src, tangential_source = tangential_source,
				c_scale = c_scale);

			# radial component for radial source
			G[:, 0, 0, 0, ℓωind] .= αr
            mul!((@view G[:, 0, 0, 1, ℓωind]), ddr, αr)

			# tangential component for radial source
			@. G[:, 1, 0, 0, ℓωind] = Ω(ℓ, 0)/(ρ*r*ω^2) * βr
            mul!((@view G[:, 1, 0, 1, ℓωind]), ddr, (@view G[:, 1, 0, 0, ℓωind]))

			if tangential_source
				# radial component for tangential source
				@. G[:, 0, 1, 0, ℓωind] = αh/√2
                mul!((@view G[:, 0, 1, 1, ℓωind]), ddr, (@view G[:, 0, 1, 0, ℓωind]))

				# tangential component for tangential source
				@. G[:, 1, 1, 0, ℓωind] = Ω(ℓ, 0)/(ρ*r*ω^2) * (βh - delta_fn)
                mul!((@view G[:, 1, 1, 1, ℓωind]), ddr, (@view G[:, 1, 1, 0, ℓωind]))
			end

			(tracker isa RemoteChannel) && put!(tracker, true)
		end

		write(file, reinterpret_as_float(G))

	end # close file

	nothing
end

function greenfn_components(r_src = r_src_default; kwargs...)
	tangential_source = get(kwargs, :tangential_source, true)
	c_scale = get(kwargs, :c_scale, 1)
	Gfn_save_directory = Gfn_path_from_source_radius(r_src, c_scale = c_scale)

	if !isdir(Gfn_save_directory)
		@info "Creating $Gfn_save_directory"
		mkdir(Gfn_save_directory)
	end

	@info "Saving output to $Gfn_save_directory"

	ℓ_arr, ω_arr = frequency_grid(Gfn_save_directory; kwargs...);
	Nν_Gfn = length(ω_arr); ω_ind_arr = 1:Nν_Gfn
	@info "Using ℓ_arr = $(ℓ_arr) and $(Nν_Gfn) frequency bins"

	modes_iter = Iterators.product(ℓ_arr, ω_ind_arr)
	num_tasks = length(modes_iter)
	num_procs = min(num_tasks, nworkers())

	append_parameters(Gfn_save_directory, num_procs = num_procs)

	tracker = RemoteChannel(() -> Channel{Bool}(num_tasks + 1))
	prog_bar = Progress(num_tasks, 2, "Green functions computed : ")

	wrapper(x) = greenfn_components_somemodes_serial_oneproc(x, r_src, c_scale, ω_arr,
	Gfn_save_directory, tracker; tangential_source = tangential_source)

	@sync begin
	 	@async begin
	 		try
	 			pmapbatch_productsplit(wrapper, ℓ_arr, ω_ind_arr);
	 			println("Finished computing Green functions")
	 		finally
	 			put!(tracker, false)
	 			finish!(prog_bar)
	 		end
	 	end
	 	while take!(tracker)
	 		 next!(prog_bar)
	 	end
	end;
	nothing # to suppress the task done message
end

function Gfn_reciprocity(; kwargs...)
	r_src = get(kwargs, :r_src, r_src_default)
	r_src_ind = radial_grid_index(r, r_src)
	r_obs = get(kwargs, :r_obs, r_obs_default)
	r_obs_ind = radial_grid_index(r, r_obs)

	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	Gfn_path_obs = Gfn_path_from_source_radius(r_obs)
	@load(joinpath(Gfn_path_src,"parameters.jld2"),
		ν_arr, Nν_Gfn, ℓ_arr, num_procs)

	num_procs_obs = get_numprocs(Gfn_path_obs)

	ℓ_range = get(kwargs, :ℓ_range, ℓ_arr)
	ν_ind_range = get(kwargs, :ν_ind_range, 1:Nν_Gfn)
	modes_iter = Iterators.product(ℓ_range, ν_ind_range)

	function summodes(ℓ_ωind_iter_on_proc)
		G_reciprocity = zeros(2, ℓ_range, ν_ind_range)

		Gfn_fits_files_src, Gfn_fits_files_obs =
		Gfn_fits_files.((Gfn_path_src, Gfn_path_obs), ((ℓ_arr, 1:Nν_Gfn),),
			(ℓ_ωind_iter_on_proc,),(num_procs, num_procs_obs))

		for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

			G10_obs_src = read_Gfn_file_at_index(
				Gfn_fits_files_src, ℓ_arr, 1:Nν_Gfn,(ℓ, ω_ind), num_procs,
				r_obs_ind, 2, 1, 1)

			G01_src_obs = read_Gfn_file_at_index(
				Gfn_fits_files_obs, ℓ_arr, 1:Nν_Gfn,(ℓ, ω_ind), num_procs_obs,
				r_src_ind, 1, 2, 1)

			G_reciprocity[1, ℓ, ω_ind] = abs(G10_obs_src)
			G_reciprocity[2, ℓ, ω_ind] = abs(G01_src_obs)

		end

		return G_reciprocity
	end

	G_reciprocity = @fetchfrom workers()[1] permutedims(
					pmapsum(summodes, modes_iter), [3, 2, 1])
end

function divG_radial!(divG::AbstractVector, ℓ::Integer, G::AbstractMatrix, drG::AbstractMatrix)
	# components in PB VSH basis
	pre = -2Ω(ℓ, 0)

	for r_ind in eachindex(divG)
		divG[r_ind] = pre * G[r_ind, 1]/r[r_ind] + drG[r_ind, 0] + 2/r[r_ind]*G[r_ind, 0]
	end

	divG
end

function divG_radial!(divG::AbstractMatrix, ℓ::Integer, G::AbstractArray{<:Any, 3}, drG::AbstractArray{<:Any, 3})
	# components in PB VSH basis
	pre = -2Ω(ℓ, 0)

	for r_ind in UnitRange(axes(divG, 2)), β in UnitRange(axes(G, 2))
		divG[β, r_ind] = pre * G[1, β, r_ind]/r[r_ind] +
		drG[0, β, r_ind] + 2/r[r_ind]*G[0, β, r_ind]
	end

	divG
end

# Radial components of dG for sound-speed perturbations
function radial_fn_δc_firstborn!(f::AbstractVector{<:Complex},
	divGsrc::AbstractVector{<:Complex}, divGobs::AbstractVector{<:Complex})

	@. f = -ρ * 2c * divGobs * divGsrc
end

function radial_fn_δc_firstborn!(f::AbstractMatrix{<:Complex},
	divGsrc::AbstractVector{<:Complex}, divGobs::AbstractMatrix{<:Complex})

	@. f = -ρ * 2c * divGobs * divGsrc
end

function radial_fn_isotropic_δc_firstborn!(f, Gsrc::AA, drGsrc::AA, divGsrc,
	Gobs::AA, drGobs::AA, divGobs, ℓ) where {AA<:AbstractArray{ComplexF64}}
	radial_fn_δc_firstborn!(f, Gsrc, drGsrc, ℓ, divGsrc, Gobs, drGobs, ℓ, divGobs)
end

# Only radial component
function radial_fn_δc_firstborn!(f::AbstractVector{<:Complex},
	Gsrc::AA, drGsrc::AA, ℓ::Integer, divGsrc::BB,
	Gobs::AA, drGobs::AA, ℓ′::Integer, divGobs::BB) where
		{AA<:AbstractArray{<:Complex, 2}, BB<:AbstractArray{<:Complex, 1}}

	# G is a tensor with one trailing vector index, the first axis is r
	# In this case the divG arrays are 1D (r)
	divG_radial!(divGobs, ℓ′, Gobs, drGobs)
	divG_radial!(divGsrc, ℓ, Gsrc, drGsrc)

	radial_fn_δc_firstborn!(f, divGsrc, divGobs)
end

# All components
function radial_fn_δc_firstborn!(f::AbstractMatrix{<:Complex},
	Gsrc::AA, drGsrc::AA, ℓ::Integer, divGsrc::BB,
	Gobs::AA, drGobs::AA, ℓ′::Integer, divGobs::BB) where
		{AA<:AbstractArray{<:Complex, 3}, BB<:AbstractArray{<:Complex, 2}}

	# G is a tensor with two vector indices
	# In this case the divG arrays are 2D (r, vec_ind)
	divG_radial!(divGobs, ℓ′, Gobs, drGobs)
	divG_radial!(divGsrc, ℓ, Gsrc, drGsrc)

	divGsrc_0 = view(divGsrc, 0, :) # Source is radial

	radial_fn_δc_firstborn!(f, divGsrc_0, divGobs)
end

# All components
# H_βαjₒjₛ(r; r₁, r₂, rₛ) = conj(f_αjₒjₛ(r, r₁, rₛ)) Gβ0jₛ(r₂, rₛ)
function Hjₒjₛω!(Hjₒjₛω_r₁r₂::AbstractArray{<:Any, 3},
	fjₒjₛ_r₁_rsrc::AbstractMatrix, Grjₛ_r₂_rsrc::AbstractVector)

	for r_ind in UnitRange(axes(fjₒjₛ_r₁_rsrc, 2)), γ in UnitRange(axes(Grjₛ_r₂_rsrc, 1))
		Gγrjₛ_r₂_rsrc = Grjₛ_r₂_rsrc[γ]
		for α in UnitRange(axes(fjₒjₛ_r₁_rsrc, 1))
			Hjₒjₛω_r₁r₂[α, γ, r_ind] = conj(fjₒjₛ_r₁_rsrc[α, r_ind]) * Gγrjₛ_r₂_rsrc
		end
	end
end

# Only radial component
# H_00jₒjₛ(r; r₁, r₂, rₛ) = conj(f_0jₒjₛ(r, r₁, rₛ)) G00jₛ(r₂, rₛ)
function Hjₒjₛω!(Hjₒjₛω_r₁r₂::AbstractVector, fjₒjₛ_r₁_rsrc::AbstractVector, Grrjₛ_r₂_rsrc)
	@. Hjₒjₛω_r₁r₂ = conj(fjₒjₛ_r₁_rsrc) * Grrjₛ_r₂_rsrc
	return Hjₒjₛω_r₁r₂
end

# Radial components of dG for flows
# Only tangential (+) component
function radial_fn_uniform_rotation_firstborn!(G::AbstractVector,
	Gsrc::AA, Gobs::AA, j, ::los_radial) where {AA<:AbstractMatrix}

	for r_ind in UnitRange(axes(Gsrc,3)),
		G[r_ind] = Gsrc[0, r_ind] * Gobs[0, r_ind] -
					Gsrc[0, r_ind] * Gobs[1, r_ind]/Ω(j, 0) -
					Gsrc[1, r_ind]/Ω(j, 0) * Gobs[0, r_ind] +
					ζjₒjₛs(j, j, 1) * Gsrc[1, r_ind] * Gobs[1, r_ind]
	end
	return G
end

function radial_fn_uniform_rotation_firstborn!(G::AbstractMatrix,
	Gsrc::AA, Gobs::AA, j, ::los_earth) where {AA<:AbstractArray{<:Any, 3}}

	for r_ind in UnitRange(axes(Gsrc,3)), α₂ in UnitRange(axes(Gobs, 2))
		G[α₂, r_ind] = Gsrc[0, 0, r_ind] * Gobs[0, α₂, r_ind] -
					Gsrc[0, 0, r_ind] * Gobs[1, α₂, r_ind]/Ω(j, 0) -
					Gsrc[1, 0, r_ind]/Ω(j, 0) * Gobs[0, α₂, r_ind] +
					ζjₒjₛs(j, j, 1) * Gsrc[1, 0, r_ind] * Gobs[1, α₂, r_ind]
	end
	return G
end

function Gⱼₒⱼₛω_u⁺_firstborn!(G::AbstractMatrix,
	Gsrc::AA, jₛ::Integer, Gobs::AA, jₒ::Integer, ::los_radial) where {AA<:AbstractMatrix}

	# The actual G¹ₗⱼ₁ⱼ₂ω_00(r, rᵢ, rₛ) =  G[:, 0] + ζ(jₛ, jₒ, ℓ)G[:, 1]
	# We store the terms separately and add them up for each ℓ

	for r_ind in UnitRange(axes(G, 2))
		G[0, r_ind] = Gsrc[0, r_ind] * Gobs[0, r_ind] -
							Gsrc[0, r_ind] * Gobs[1, r_ind]/Ω(jₒ, 0) -
							Gsrc[1, r_ind]/Ω(jₛ, 0) * Gobs[0, r_ind]

		G[1, r_ind] = Gsrc[1, r_ind] * Gobs[1, r_ind]
	end
	return G
end

function Gⱼₒⱼₛω_u⁺_firstborn!(G::AbstractArray{<:Any, 3},
	Gsrc::AA, jₛ::Integer, Gobs::AA, jₒ::Integer,
	::los_earth) where {AA<:AbstractArray{<:Any, 3}}

	# The actual G¹ₗⱼ₁ⱼ₂ω_α₁0(r, rᵢ, rₛ) = G[:, 0, α₁] + ζ(jₛ, jₒ, ℓ)G[:, 1, α₁]
	# We store the terms separately and add them up for each ℓ

	for r_ind in UnitRange(axes(G, 3)), α₁ in UnitRange(axes(G, 2))
		G[0, α₁, r_ind] = Gsrc[0, 0, r_ind] * Gobs[0, α₁, r_ind] -
							Gsrc[0, 0, r_ind] * Gobs[1, α₁, r_ind]/Ω(jₒ, 0) -
							Gsrc[1, 0, r_ind]/Ω(jₛ, 0) * Gobs[0, α₁, r_ind]

		G[1, α₁, r_ind] = Gsrc[1, 0, r_ind] * Gobs[1, α₁, r_ind]
	end
	return G
end

function Gⱼₒⱼₛω_u⁰_firstborn!(G::AbstractMatrix{<:Any},
	drGsrc::AA, jₛ::Integer, Gobs::AA, jₒ::Integer, ::los_radial) where {AA<:AbstractMatrix}

	# The actual G⁰ₗⱼ₁ⱼ₂ω_00(r, rᵢ, rₛ) =  G[:, 0] + ζ(jₛ, jₒ, ℓ)G[:, 1]
	# We store the terms separately and add them up for each ℓ

	@. G = Gobs * drGsrc
	return G
end

function Gⱼₒⱼₛω_u⁰_firstborn!(G::AbstractArray{<:Any, 3},
	drGsrc::AA, jₛ::Integer, Gobs::AA, jₒ::Integer, ::los_earth) where {AA<:AbstractArray{<:Any, 3}}

	# The actual G⁰ₗⱼ₁ⱼ₂ω_α₁0(r, rᵢ, rₛ) = G[:, 0, α₁] + ζ(jₛ, jₒ, ℓ)G[:, 1, α₁]
	# We store the terms separately and add them up for each ℓ
	for r_ind in UnitRange(axes(Gobs, 3)), α₁ in UnitRange(axes(Gobs, 2)), ind in UnitRange(axes(G,1))
		G[ind, α₁, r_ind] = Gobs[ind, α₁, r_ind] * drGsrc[ind, 0, r_ind]
	end
	return G
end

# This function computes Gparts
# Components (0) and (+)
function Gⱼₒⱼₛω_u⁰⁺_firstborn!(G, Gsrc, drGsrc, jₛ::Integer, Gobs, jₒ::Integer, los::los_direction)
	Gⱼₒⱼₛω_u⁰_firstborn!(view(G, .., 0), drGsrc, jₛ, Gobs, jₒ, los)
	Gⱼₒⱼₛω_u⁺_firstborn!(view(G, .., 1),  Gsrc, jₛ, Gobs, jₒ, los)
	return G
end

# We evaluate Gγₗⱼ₁ⱼ₂ω_00(r, rᵢ, rₛ) = Gsum[r,γ] = G[:, 0] + ζ(jₛ, jₒ, ℓ)G[:, 1]
# This function is run once for each ℓ using cached values of G
function Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁::StructArray{<:Complex, 2}, Gparts::StructArray{<:Complex, 3}, jₒ, jₛ, ℓ)

	coeff = ζjₒjₛs( jₒ, jₛ, ℓ)

	for γ in UnitRange(axes(Gγₗⱼₒⱼₛω_α₁, 2)), rind in UnitRange(axes(Gγₗⱼₒⱼₛω_α₁, 1))
		Gγₗⱼₒⱼₛω_α₁.re[rind, γ] = Gparts.re[0, rind, γ] + coeff * Gparts.re[1, rind, γ]
		Gγₗⱼₒⱼₛω_α₁.im[rind, γ] = Gparts.im[0, rind, γ] + coeff * Gparts.im[1, rind, γ]
	end
	return Gγₗⱼₒⱼₛω_α₁
end

# We evaluate Gγₗⱼ₁ⱼ₂ω_α₁0(r, rᵢ, rₛ) = Gsum[r, γ, α₁] = G[:, 0, α₁] + ζ(jₛ, jₒ, ℓ)G[:, 1, α₁]
# This function is run once for each ℓ using cached values of G
function Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁::StructArray{<:Complex, 3}, Gparts::StructArray{<:Complex, 4}, jₒ, jₛ, ℓ)

	coeff = ζjₒjₛs( jₒ, jₛ, ℓ)

	GT = HybridArray{Tuple{2,StaticArrays.Dynamic(),2}}
	GR = GT(parent(Gγₗⱼₒⱼₛω_α₁.re))
	GI = GT(parent(Gγₗⱼₒⱼₛω_α₁.im))

	GPT = HybridArray{Tuple{2,2,StaticArrays.Dynamic(),2}}
	GPR = GPT(parent(Gparts.re))
	GPI = GPT(parent(Gparts.im))

	@turbo for I in CartesianIndices(GR)
		GR[I] = GPR[1, I] + coeff * GPR[2, I]
		GI[I] = GPI[1, I] + coeff * GPI[2, I]
	end
	return Gγₗⱼₒⱼₛω_α₁
end

# We compute Hγₗⱼ₁ⱼ₂ω_00(r, r₁, r₂, rₛ) = conj(Gγₗⱼ₁ⱼ₂ω_00(r, r₁, rₛ)) * Gⱼ₂ω_00(r₂, rₛ)
function Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(H::StructArray{<:Complex, 2}, Gparts,
	Gγₗⱼₒⱼₛω_α₁::AbstractMatrix{<:Complex}, Grr::Complex, jₒ, jₛ, ℓ)

	Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁, Gparts, jₒ, jₛ, ℓ)
	# @. H = conj(Gγₗⱼₒⱼₛω_α₁) * Grr
	@. H.re = Gγₗⱼₒⱼₛω_α₁.re * real(Grr) + Gγₗⱼₒⱼₛω_α₁.im * imag(Grr)
	@. H.im = Gγₗⱼₒⱼₛω_α₁.re * imag(Grr) - Gγₗⱼₒⱼₛω_α₁.im * real(Grr)
	return H
end

# We compute Hγₗⱼ₁ⱼ₂ω_α₁α₂(r, r₁, r₂, rₛ) = conj(Gγₗⱼ₁ⱼ₂ω_α₁0(r, r₁, rₛ)) * Gⱼ₂ω_α₂0(r₂, rₛ)
function Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(H::StructArray{<:Complex, 4}, Gparts,
	Gγₗⱼₒⱼₛω_α₁_r₁::StructArray{<:Complex, 3}, G::AbstractVector{<:Complex}, jₒ, jₛ, ℓ)

	Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁_r₁, Gparts, jₒ, jₛ, ℓ)

	HT = HybridArray{Tuple{2,2,StaticArrays.Dynamic(),2}}
	HR = HT(parent(H.re))
	HI = HT(parent(H.im))

	GT = HybridArray{Tuple{2,StaticArrays.Dynamic(),2}}
	GR = GT(parent(Gγₗⱼₒⱼₛω_α₁_r₁.re))
	GI = GT(parent(Gγₗⱼₒⱼₛω_α₁_r₁.im))

	GsrcR = SVector{2}(real(G[0]), real(G[1]))
	GsrcI = SVector{2}(imag(G[0]), imag(G[1]))

	@turbo for γ in UnitRange(axes(HR, 4)), rind in UnitRange(axes(HR, 3)), obsind2 in UnitRange(axes(HR, 2))
		GsrcR_obs2 = GsrcR[obsind2]
		GsrcI_obs2 = GsrcI[obsind2]
		for obsind1 in UnitRange(axes(HR, 1))
			HR[obsind1, obsind2, rind, γ] = GR[obsind1, rind, γ] * GsrcR_obs2 + GI[obsind1, rind, γ] * GsrcI_obs2
			HI[obsind1, obsind2, rind, γ] = GR[obsind1, rind, γ] * GsrcI_obs2 - GI[obsind1, rind, γ] * GsrcR_obs2
		end
	end
	return H
end

function permuteGfn(path)
	newdir = joinpath(path, "..", path*"_flipped")
	isdir(newdir) || mkdir(newdir)
	files =	filter(x -> endswith(x, ".fits"), readdir(path))
	@showprogress 1 for filename in files
		_data = FITSIO.fitsread(joinpath(path, filename))::Array{Float64,5}
		data = reshape(_data, 2, nr, 2, 2, 2, :)
		data2 = permutedims(data, [1, 3, 4, 2, 5, 6])
		FITSIO.fitswrite(joinpath(newdir, filename), data2)
	end
	cp(joinpath(path, "parameters.jld2"), joinpath(newdir, "parameters.jld2"), force = true)
end
