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

	v = ParallelUtilities.whichproc_localindex((ℓ_arr, ω_ind_arr), ℓω, num_procs)
	v === nothing && error("could not locate ℓω = $ℓω")
	proc_id_mode, ℓω_index_in_file = v
	G_file = G_fits_dict[proc_id_mode].hdu
	read_Gfn_file_at_index!(G, G_file, I..., ℓω_index_in_file)
	return G
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

	return Rsun, nr, r, dr, ddr, c,ρ, g, N2, γ_damping
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



Ω(ℓ, N) = √((ℓ+N)*(ℓ-N+1)/2)
ζjₒjₛs(j₁, j₂, ℓ) = (Ω(j₁, 0)^2 + Ω(j₂, 0)^2 - Ω(ℓ, 0)^2)/(Ω(j₁, 0)*Ω(j₂, 0))
Njₒjₛs(jₒ, jₛ, s) = √((2jₒ+1)*(2jₛ+1)/(4π*(2s+1)))

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
# We solve in the Hansen VSH basis first and change over to the PB basis

function solve_for_components!(M, S, α, β)
	H = M\S # solve the equation

	@views @. α = H[1:nr]
	@views @. β = H[nr+1:end]
    return α, β
end

function source!(Sh, ω, ℓ, delta_fn)
    @views @. Sh[nr+1:end-1] = -√(ℓ*(ℓ+1))/ω^2 * delta_fn[1:nr-1] /(r[1:nr-1]*ρ[1:nr-1])
	Sh[end] = 0
    return Sh;
end

function ℒr!(L12, L21, c′; stencil_gridpts = Dict(6=>3, 42=>5), kw...)
    # Dirichlet condition β(r_out)=0 on the right edge
    dbydr!(L12, dr, stencil_gridpts = stencil_gridpts, left_edge_npts = 2, right_edge_npts = 2)
	L12[1:2] .= 0 # dirichlet boundary condition on G at the inner boundary

    dinds = diagind(L12)
    @. L12[dinds] += g/c′^2

    # Right edge ghost to use point α(r_out) in computing derivatives
    # Dirichlet condition α(r_in)=0 on the left edge
    dbydr!(L21, dr, stencil_gridpts = stencil_gridpts, left_edge_npts = 2, right_edge_npts = 2)
	L21[end-1:end] .= 0

    dinds = diagind(L21)
    @. L21[dinds] += 2/r - g/c′^2

	return L12, L21
end

function wave_operator_diagonal!(Mdiag, Mdiagfixed, ω, ℓ, c′, invρr²; kw...)
	ωC = ω - im * γ_damping(ω)

	Mdiag .= Mdiagfixed

	for (dind, rind) in zip(1:nr, 1:nr)
		Mdiag[dind] += -ωC^2 * ρ[rind]
	end
	for (dind, rind) in zip(nr+1:2nr, 1:nr)
		Mdiag[dind] += -ℓ*(ℓ+1) / ωC^2 * invρr²[rind]
	end

	return Mdiag
end

function greenfn_components_onemode!(ω, ℓ, αrℓω, βrℓω, αhℓω, βhℓω, M_ωℓ, Mdiagfixed, Mdiag,
	Sr, Sh, delta_fn, c′, invρr²; kwargs...)

	tangential_source = get(kwargs, :tangential_source, true)

	source!(Sh, ω, ℓ, delta_fn)

	wave_operator_diagonal!(Mdiag, Mdiagfixed, ω, ℓ, c′, invρr²; kwargs...);
	M_ωℓ[diagind(M_ωℓ)] .= Mdiag

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
	r_src, c_scale, ω_arr,
	tangential_source::Bool = true,
	tracker::Union{Nothing, RemoteChannel} = nothing,
	)

	# save real and imaginary parts separately
	β = 0:1
	γ = tangential_source ? (0:1) : (0:0)
	G = zeros(ComplexF64, nr, β, γ, 0:1, length(ℓ_ωind_proc))

	r_src_on_grid = radial_grid_closest(r, r_src)
	σsrc = σsrc_grid(r_src_on_grid)
	delta_fn = delta_fn_gaussian_approx(r_src_on_grid, σsrc)
	Sr, Sh = zeros(2nr), zeros(2nr);
	@views @. Sr[2:nr] = delta_fn[2:nr];

	αr, βr = zeros(ComplexF64, nr), zeros(ComplexF64, nr);
	αh, βh = zeros(ComplexF64, nr), zeros(ComplexF64, nr);

	# temporary arrys used to compute the derivative operator
	M_ωℓ = zeros(ComplexF64, 2nr, 2nr);
	L12 = zeros(nr, nr);
	L21 = zero(L12);
	Mdiagfixed = zeros(2nr); # the terms that do not depend on ω and ℓ
	Mdiag = zeros(ComplexF64, 2nr); # the terms that do not depend on ω and ℓ

	stencil_gridpts = Dict(6=>3, 42=>5);

	if c_scale != 1
		c′= c .* c_scale
		N2′= @. N2 + g^2/c^2 * (1-1/c_scale^2)
	else
		c′ = c;
		N2′ = N2;
	end
	invρr² = @. 1 / (ρ * r^2);

	@. Mdiagfixed[1:nr] = ρ * N2′;
	@. Mdiagfixed[nr+1:end] = 1 / (ρ * c′^2);

	Gωℓ, drGωℓ = zeros(ComplexF64, nr), zeros(ComplexF64, nr)

	ℒr!(L12, L21, c′, stencil_gridpts = stencil_gridpts)

	# top right, ∂ᵣβ, (nr x nr)
	M_ωℓ[1:nr, nr+1:end] = L12

	# bottom left ∂ᵣα, (nr x nr)
	M_ωℓ[nr+1:end, 1:nr] = L21

	for (ℓωind, (ℓ, ω_ind)) in enumerate(ℓ_ωind_proc)

		ω = ω_arr[ω_ind]

		greenfn_components_onemode!(ω, ℓ, αr,
			βr, αh, βh, M_ωℓ, Mdiagfixed, Mdiag, Sr, Sh, delta_fn,
			c′, invρr²,
			r_src = r_src, tangential_source = tangential_source,
			c_scale = c_scale);

		# radial component for radial source
		@. G[:, 0, 0, 0, ℓωind] = αr

		# tangential component for radial source
		@. G[:, 1, 0, 0, ℓωind] = Ω(ℓ, 0)/(ρ*r*ω^2) * βr

		if tangential_source
			# radial component for tangential source
			@. G[:, 0, 1, 0, ℓωind] = αh/√2

			# tangential component for tangential source
			@. G[:, 1, 1, 0, ℓωind] = Ω(ℓ, 0)/(ρ*r*ω^2) * (βh - delta_fn)
		end

		for srcind in UnitRange(axes(G,3)), obsind in UnitRange(axes(G,2))
			Gωℓ .= @view G[:, obsind, srcind, 0, ℓωind]
			mul!(drGωℓ, ddr, Gωℓ)
			G[:, obsind, srcind, 1, ℓωind] = drGωℓ
		end

		(tracker isa RemoteChannel) && put!(tracker, true)
	end

	permutedims(reshape(reinterpret_as_float(G), 2, nr, 2, 2, 2, :), [1, 3, 4, 2, 5, 6])
end

function greenfn_components_somemodes_serial_oneproc_fits(ℓ_ωind_proc::ProductSplit,
	r_src, c_scale,
	ω_arr, Gfn_save_directory,
	tangential_source::Bool = true,
	tracker::Union{Nothing, RemoteChannel} = nothing
    )

	rank = ParallelUtilities.workerrank(ℓ_ωind_proc)
	save_path = joinpath(Gfn_save_directory, @sprintf "Gfn_proc_%03d.fits" rank)

	FITS(save_path, "w") do file
		GR = greenfn_components_somemodes_serial_oneproc(
			ℓ_ωind_proc, r_src, c_scale, ω_arr, tangential_source, tracker)

		write(file, GR)
	end

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

	wrapper(x) = greenfn_components_somemodes_serial_oneproc_fits(x, r_src, c_scale, ω_arr,
	Gfn_save_directory, tangential_source, tracker)

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

function Gfn_reciprocity_partial(localtimer, ℓ_ωind_iter_on_proc, r_src, r_obs)
	ℓ_arr, ν_arr = ParallelUtilities.getiterators(ℓ_ωind_iter_on_proc)
	Nν_Gfn = length(ν_arr)

	r_src_ind = radial_grid_index(r, r_src)
	r_obs_ind = radial_grid_index(r, r_obs)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	Gfn_path_obs = Gfn_path_from_source_radius(r_obs)
	num_procs_src = get_numprocs(Gfn_path_src)
	num_procs_obs = get_numprocs(Gfn_path_obs)

	G_reciprocity = zeros(2, ℓ_arr, ν_arr)

	Gfn_fits_files_src, Gfn_fits_files_obs =
		Gfn_fits_files.((Gfn_path_src, Gfn_path_obs), ((ℓ_arr, 1:Nν_Gfn),),
			(ℓ_ωind_iter_on_proc,), (num_procs_src, num_procs_obs))

	G01 = zeros(ComplexF64)
	G10 = zeros(ComplexF64)

	for (ℓ, ω_ind) in ℓ_ωind_iter_on_proc

		read_Gfn_file_at_index!(G10,
			Gfn_fits_files_src, (ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), num_procs_src,
			2, 1, r_obs_ind, 1)

		read_Gfn_file_at_index!(G01,
			Gfn_fits_files_obs, (ℓ_arr, 1:Nν_Gfn), (ℓ, ω_ind), num_procs_obs,
			1, 2, r_src_ind, 1)

		G_reciprocity[1, ℓ, ω_ind] = abs(G10[])
		G_reciprocity[2, ℓ, ω_ind] = abs(G01[])
	end

	return parent(G_reciprocity)
end

function Gfn_reciprocity(comm; kwargs...)
	r_src = get(kwargs, :r_src, r_src_default)
	r_obs = get(kwargs, :r_obs, r_obs_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load(joinpath(Gfn_path_src,"parameters.jld2"),
		ν_arr, Nν_Gfn, ℓ_arr, num_procs)

	ℓ_range = get(kwargs, :ℓ_range, ℓ_arr)
	ν_ind_range = get(kwargs, :ν_ind_range, 1:Nν_Gfn)

	G_reciprocity = pmapsum(comm, Gfn_reciprocity_partial, (ℓ_range, ν_ind_range), r_src, r_obs)
	permutedims(G_reciprocity, [3,2,1])
end

#################################################################################
# Functions that are used in computing the kernels
#################################################################################

_mulR(a::Complex, b::Complex) = real(a)*real(b) - imag(a)*imag(b)
_mulI(a::Complex, b::Complex) = real(a)*imag(b) + imag(a)*real(b)

_structarrayparent(S::StructArray{<:Complex}) = StructArray{eltype(S)}(map(parent, (S.re, S.im)))
_structarrayparent(S::StructArray{<:Complex}, ::Type{T}) where {T} = StructArray{eltype(S)}(map(T∘parent, (S.re, S.im)))
_structarrayparent(S::AbstractArray, ::Type{T}) where {T} = T(no_offset_view(parent(S)))
_structarray(S::StructArray{<:Complex}, ::Type{T}) where {T} = StructArray{eltype(S)}(map(T, (S.re, S.im)))
_structarray(S::AbstractArray, ::Type{T}) where {T} = T(no_offset_view(S))

function divG_radial!(divG::AbstractVector, ℓ::Integer, G::AbstractMatrix, drG::AbstractMatrix)
	# components in PB VSH basis
	pre = -2Ω(ℓ, 0)

	T = HybridArray{Tuple{2,StaticArrays.Dynamic()}}
	GP = T(parent(G))
	drGP = T(parent(drG))

	@turbo for r_ind in eachindex(divG)
		divG[r_ind] = pre * GP[2, r_ind]/r[r_ind] + drGP[1, r_ind] + 2/r[r_ind]*GP[1, r_ind]
	end

	divG
end

function divG_radial!(divG::AbstractMatrix, ℓ::Integer, G::AbstractArray{<:Any, 3}, drG::AbstractArray{<:Any, 3})
	# components in PB VSH basis
	pre = -2Ω(ℓ, 0)

	Tdiv = HybridArray{Tuple{2,StaticArrays.Dynamic()}}
	divGP = Tdiv(parent(divG))

	TG = HybridArray{Tuple{2,2,StaticArrays.Dynamic()}}
	GP = TG(parent(G))
	drGP = TG(parent(drG))

	@turbo for r_ind in UnitRange(axes(divG, 2))
		invr = 1/r[r_ind]
		for β in UnitRange(axes(GP, 2))
			divGP[β, r_ind] = pre * GP[2, β, r_ind]*invr + drGP[1, β, r_ind] + 2invr*GP[1, β, r_ind]
		end
	end

	divG
end

# Radial components of dG for sound-speed perturbations
function radial_fn_δc_firstborn!(f::StructArray{<:Complex,1},
	divGsrc::AbstractVector{<:Complex}, divGobs::AbstractVector{<:Complex})

	@turbo for I in eachindex(f)
		pre = -ρ[I] * 2c[I]
		f.re[I] = pre * _mulR(divGobs[I], divGsrc[I])
		f.im[I] = pre * _mulI(divGobs[I], divGsrc[I])
	end
	return f
end

function radial_fn_δc_firstborn!(f::AbstractMatrix{<:Complex},
	divGsrc::AbstractVector{<:Complex}, divGobs::AbstractMatrix{<:Complex})

	HT = HybridArray{Tuple{2,StaticArrays.Dynamic()}}
	divGobsH = HT(parent(divGobs))

	fR = HT(parent(f.re))
	fI = HT(parent(f.im))

	@turbo for r_ind in eachindex(r)
		twoρcdivGsrc = -ρ[r_ind] * 2c[r_ind] * divGsrc[r_ind]
		for obsind in axes(fR,1)
			fR[obsind, r_ind] = _mulR(divGobsH[obsind, r_ind], twoρcdivGsrc)
			fI[obsind, r_ind] = _mulI(divGobsH[obsind, r_ind], twoρcdivGsrc)
		end
	end
	return f
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

	HT = HybridArray{Tuple{2,2,StaticArrays.Dynamic()}}
	HP = _structarrayparent(Hjₒjₛω_r₁r₂, HT)

	FT = HybridArray{Tuple{2,StaticArrays.Dynamic()}}
	FP = _structarrayparent(fjₒjₛ_r₁_rsrc, FT)

	Grjₛ_r₂_rsrcP = no_offset_view(Grjₛ_r₂_rsrc)

	@turbo for r_ind in axes(HP, 3), γ in axes(HP, 2)
		Gγrjₛ_r₂_rsrc = Grjₛ_r₂_rsrcP[γ]
		for α in axes(HP, 1)
			HP[α, γ, r_ind] = conj(FP[α, r_ind]) * Gγrjₛ_r₂_rsrc
		end
	end
	return Hjₒjₛω_r₁r₂
end

# Only radial component
# H_00jₒjₛ(r; r₁, r₂, rₛ) = conj(f_0jₒjₛ(r, r₁, rₛ)) G00jₛ(r₂, rₛ)
function Hjₒjₛω!(Hjₒjₛω_r₁r₂::AbstractVector, fjₒjₛ_r₁_rsrc::AbstractVector, Grrjₛ_r₂_rsrc)
	@turbo for I in CartesianIndices(Hjₒjₛω_r₁r₂)
		Hjₒjₛω_r₁r₂[I] = conj(fjₒjₛ_r₁_rsrc[I]) * Grrjₛ_r₂_rsrc
	end
	return Hjₒjₛω_r₁r₂
end

# Radial components of dG for flows
# Only tangential (+) component
function radial_fn_uniform_rotation_firstborn!(G::AbstractVector,
	Gsrc::AA, Gobs::AA, j, ::los_radial) where {AA<:AbstractMatrix}

	for r_ind in UnitRange(axes(Gsrc,2))
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

function Gⱼₒⱼₛω_u⁺_firstborn!(Gparts⁺, Gsrc, jₛ, Gobs, jₒ, ::los_radial)

	# The actual G¹ₗⱼ₁ⱼ₂ω_00(r, rᵢ, rₛ) =  G[:, 0] + ζ(jₛ, jₒ, ℓ)G[:, 1]
	# We store the terms separately and add them up for each ℓ

	GT = HybridArray{Tuple{2,StaticArrays.Dynamic()}}
	GsrcH = GT(parent(Gsrc))
	GobsH = GT(parent(Gobs))

	GH = _structarrayparent(Gparts⁺, GT)

	Ωjₒ0 = Ω(jₒ, 0)
	Ωjₛ0 = Ω(jₛ, 0)

	@turbo for r_ind in axes(GH, 2)
		GH[1, r_ind] = GsrcH[1, r_ind] * GobsH[1, r_ind] -
						GsrcH[1, r_ind] * GobsH[2, r_ind]/Ωjₒ0 -
						GsrcH[2, r_ind] * GobsH[1, r_ind]/Ωjₛ0

		GH[2, r_ind] = GsrcH[2, r_ind] * GobsH[2, r_ind]
	end
	return Gparts⁺
end

function Gⱼₒⱼₛω_u⁺_firstborn!(Gparts⁺, Gsrc, jₛ, Gobs, jₒ, ::los_earth)

	# The actual G¹ₗⱼ₁ⱼ₂ω_α₁0(r, rᵢ, rₛ) = G[:, 0, α₁] + ζ(jₛ, jₒ, ℓ)G[:, 1, α₁]
	# We store the terms separately and add them up for each ℓ

	GT = HybridArray{Tuple{2,2,StaticArrays.Dynamic()}}
	GsrcH = GT(parent(Gsrc))
	GobsH = GT(parent(Gobs))

	GH = _structarrayparent(Gparts⁺, GT)

	Ωjₒ0 = Ω(jₒ, 0)
	Ωjₛ0 = Ω(jₛ, 0)

	@turbo for r_ind in axes(GH, 3), α₁ in axes(GH, 2)
		GH[1, α₁, r_ind] = GsrcH[1, 1, r_ind] * GobsH[1, α₁, r_ind] -
							GsrcH[1, 1, r_ind] * GobsH[2, α₁, r_ind]/Ωjₒ0 -
							GsrcH[2, 1, r_ind] * GobsH[1, α₁, r_ind]/Ωjₛ0

		GH[2, α₁, r_ind] = GsrcH[2, 1, r_ind] * GobsH[2, α₁, r_ind]
	end
	return Gparts⁺
end

function Gⱼₒⱼₛω_u⁰_firstborn!(Gparts⁰, drGsrc, jₛ, Gobs, jₒ, ::los_radial)

	# The term corresponding to u⁰ is G⁰₀ ∂ᵣG⁰₀ + ζ(jₛ, jₒ, ℓ) G¹₀ ∂ᵣG¹₀
	# We express this as G⁰ₗⱼ₁ⱼ₂ω_00(r, rᵢ, rₛ) =  G[:, 0] + ζ(jₛ, jₒ, ℓ)G[:, 1]
	# We store the terms separately and add them up for each ℓ

	GT = HybridArray{Tuple{2,StaticArrays.Dynamic()}}
	drGsrcH = GT(parent(drGsrc))
	GobsH = GT(parent(Gobs))

	GH = _structarrayparent(Gparts⁰, GT)

	@turbo for I in CartesianIndices(GH)
		GH[I] = GobsH[I] * drGsrcH[I]
	end
	return Gparts⁰
end

function Gⱼₒⱼₛω_u⁰_firstborn!(Gparts⁰, drGsrc, jₛ, Gobs, jₒ, ::los_earth)

	# The term corresponding to u⁰ is G⁰α₁ ∂ᵣG⁰₀ + ζ(jₛ, jₒ, ℓ) G¹α₁ ∂ᵣG¹₀
	# We express this as G⁰ₗⱼ₁ⱼ₂ω_α₁0(r, rᵢ, rₛ) = G[0, α₁, :] + ζ(jₛ, jₒ, ℓ)G[1, α₁, :]
	# We store the terms separately and add them up for each ℓ

	GT = HybridArray{Tuple{2,2,StaticArrays.Dynamic()}}
	GobsH = GT(parent(Gobs))
	drGsrcH = GT(parent(drGsrc))

	GH = _structarrayparent(Gparts⁰, GT)

	@turbo for r_ind in axes(GH, 3), α₁ in axes(GH, 2), ind in axes(GH, 1)
		GH[ind, α₁, r_ind] = GobsH[ind, α₁, r_ind] * drGsrcH[ind, 1, r_ind]
	end
	return Gparts⁰
end

# This function computes Gparts
# Components (0) and (+)
function Gⱼₒⱼₛω_u⁰⁺_firstborn!(Gparts, Gparts2, Gsrc, drGsrc, jₛ, Gobs, jₒ, los::los_direction)
	Gⱼₒⱼₛω_u⁰_firstborn!(Gparts[0], drGsrc, jₛ, Gobs, jₒ, los)
	Gⱼₒⱼₛω_u⁺_firstborn!(Gparts[1],  Gsrc, jₛ, Gobs, jₒ, los)
	copyto!(Gparts2, 1, Gparts[0], 1, length(Gparts[0]))
	copyto!(Gparts2, length(Gparts[0]) + 1, Gparts[1], 1, length(Gparts[1]))
	return Gparts2
end

# We evaluate Gγₗⱼ₁ⱼ₂ω_00(r, rᵢ, rₛ) = Gsum[γ, r] = Gparts[0, :] + ζ(jₛ, jₒ, ℓ)G[1, :]
# This function is run once for each ℓ using cached values of G
function Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁::AbstractMatrix, Gparts2, jₒ, jₛ, ℓ)

	coeff = ζjₒjₛs( jₒ, jₛ, ℓ)

	GT = HybridArray{Tuple{StaticArrays.Dynamic(),2}}
	G = _structarrayparent(Gγₗⱼₒⱼₛω_α₁, GT)

	GPT = HybridArray{Tuple{2,StaticArrays.Dynamic(),2}}
	GP = _structarrayparent(Gparts2, GPT)

	@turbo for I in CartesianIndices(G)
		G[I] = GP[1, I] + coeff * GP[2, I]
	end
	return Gγₗⱼₒⱼₛω_α₁
end

# We evaluate Gγₗⱼ₁ⱼ₂ω_α₁0(r, rᵢ, rₛ) = Gsum[r, γ, α₁] = G[:, 0, α₁] + ζ(jₛ, jₒ, ℓ)G[:, 1, α₁]
# This function is run once for each ℓ using cached values of G
function Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁::StructArray{<:Complex, 3}, Gparts2, jₒ, jₛ, ℓ)

	coeff = ζjₒjₛs( jₒ, jₛ, ℓ)

	GT = HybridArray{Tuple{2,StaticArrays.Dynamic(),2}}
	G = _structarrayparent(Gγₗⱼₒⱼₛω_α₁, GT)

	GPT = HybridArray{Tuple{2,2,StaticArrays.Dynamic(),2}}
	GP = _structarrayparent(Gparts2, GPT)

	@turbo for I in CartesianIndices(G)
		G[I] = GP[1, I] + coeff * GP[2, I]
	end
	return Gγₗⱼₒⱼₛω_α₁
end

# We compute Hγₗⱼ₁ⱼ₂ω_00(r, r₁, r₂, rₛ) = conj(Gγₗⱼ₁ⱼ₂ω_00(r, r₁, rₛ)) * Gⱼ₂ω_00(r₂, rₛ)
function Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(H::StructArray{<:Complex, 2}, Gparts,
	Gγₗⱼₒⱼₛω_α₁::AbstractMatrix{<:Complex}, Grr::Complex, jₒ, jₛ, ℓ)

	HT = HybridArray{Tuple{StaticArrays.Dynamic(),2}}
	HR = HT(parent(H.re))
	HI = HT(parent(H.im))

	GT = HybridArray{Tuple{StaticArrays.Dynamic(),2}}
	GR = GT(parent(Gγₗⱼₒⱼₛω_α₁.re))
	GI = GT(parent(Gγₗⱼₒⱼₛω_α₁.im))

	Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁, Gparts, jₒ, jₛ, ℓ)
	# @. H = conj(Gγₗⱼₒⱼₛω_α₁) * Grr
	@turbo for I in CartesianIndices(HR)
		HR[I] = GR[I] * real(Grr) + GI[I] * imag(Grr)
		HI[I] = GR[I] * imag(Grr) - GI[I] * real(Grr)
	end
	return H
end

# We compute Hγₗⱼ₁ⱼ₂ω_α₁α₂(r, r₁, r₂, rₛ) = conj(Gγₗⱼ₁ⱼ₂ω_α₁0(r, r₁, rₛ)) * Gⱼ₂ω_α₂0(r₂, rₛ)
function Hγₗⱼₒⱼₛω_α₁α₂_firstborn!(H::StructArray{<:Complex, 4}, Gparts,
	Gγₗⱼₒⱼₛω_α₁::StructArray{<:Complex, 3}, G::AbstractVector{<:Complex}, jₒ, jₛ, ℓ)

	Gγₗⱼₒⱼₛω_α₁_firstborn!(Gγₗⱼₒⱼₛω_α₁, Gparts, jₒ, jₛ, ℓ)

	HT = HybridArray{Tuple{2,2,StaticArrays.Dynamic(),2}}
	HS = _structarrayparent(H, HT)

	GT = HybridArray{Tuple{2,StaticArrays.Dynamic(),2}}
	Gγₗⱼₒⱼₛω_α₁sH = _structarrayparent(Gγₗⱼₒⱼₛω_α₁, GT)

	Gp = parent(G)

	@turbo for γ in axes(HS, 4), rind in axes(HS, 3), obsind2 in axes(HS, 2)
		Gsrc_obs2 = Gp[obsind2]
		for obsind1 in axes(HS, 1)
			HS[obsind1, obsind2, rind, γ] = conj(Gγₗⱼₒⱼₛω_α₁sH[obsind1, rind, γ]) * Gsrc_obs2
		end
	end
	return H
end
