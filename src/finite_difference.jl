###########################################################################################
# Operators
###########################################################################################
const DerivStencilScratch = Dict{NTuple{2,Int}, Matrix{Float64}}()

function derivStencil(order::Integer, ptsleft::Integer, ptsright::Integer; gridspacing::Real = 1)

	invM = get!(DerivStencilScratch, (ptsleft, ptsright)) do
		M = zeros(ptsleft + ptsright + 1, ptsleft + ptsright + 1)
		for (m_ind, m) in enumerate(-ptsleft:ptsright), (p_ind, p) in enumerate(0:(ptsleft + ptsright))
			M[p_ind, m_ind] = m^p/factorial(p)
		end
		inv(M)
	end

	M_order = invM[:, order + 1]
	if (ptsleft == ptsright) && isodd(order)
		# Fix loss of numerical precision
		M_order[ptsleft + 1] = 0
	end

	den = gridspacing^order
	M_order ./= den
	return M_order
end

function derivStencil(order::Integer, N::Integer; kwargs...)
	# given N points decide on the number of points to the left and to the right
	if isodd(N)
		return derivStencil(order, div(N, 2), div(N, 2); kwargs...)
	else
		return derivStencil(order, div(N, 2)-1, div(N, 2); kwargs...)
	end
end

function derivStencil!(v, order::Integer, ptsleft::Integer, ptsright::Integer; gridspacing::Real = 1)

	invM = get!(DerivStencilScratch, (ptsleft, ptsright)) do
		M = zeros(ptsleft + ptsright + 1, ptsleft + ptsright + 1)
		for (m_ind, m) in enumerate(-ptsleft:ptsright), (p_ind, p) in enumerate(0:(ptsleft + ptsright))
			M[p_ind, m_ind] = m^p/factorial(p)
		end
		inv(M)
	end

	v .= @view invM[:, order + 1]
	if (ptsleft == ptsright) && isodd(order)
		# Fix loss of numerical precision
		v[ptsleft + 1] = 0
	end

	den = gridspacing^order
	v ./= den
	return v
end

function derivStencil!(v, order::Integer, N::Integer; kwargs...)
	# given N points decide on the number of points to the left and to the right
	if isodd(N)
		return derivStencil!(v, order, div(N, 2), div(N, 2); kwargs...)
	else
		return derivStencil!(v, order, div(N, 2)-1, div(N, 2); kwargs...)
	end
end

nextodd(n::Integer) = n + 1 + isodd(n)

function ceilsearch(arr, n)
	# assume sorted
	for elem in arr
		if elem>=n
			return elem
		end
	end
	return n
end

function D(N; stencil_gridpts = Dict(6=>3, 42=>5), #= Dictionary of gridpt => order of derivative upto that gridpt =#
	left_edge_npts = 2, left_edge_ghost = false,
	right_edge_npts = 2, right_edge_ghost = false,
	left_Dirichlet = false,
	right_Dirichlet = false,
	kwargs...)

	@assert(N≥2,"Need at least 2 points to compute the derivative")

	N_cols = N
	if left_edge_ghost
		N_cols += 1
	end
	if right_edge_ghost
		N_cols += 1
	end

	S = zeros(N, N_cols)

	D!(S, N; stencil_gridpts = stencil_gridpts,
		left_edge_npts = left_edge_npts, left_edge_ghost = left_edge_ghost,
		right_edge_npts = right_edge_npts, right_edge_ghost = right_edge_ghost,
		left_Dirichlet = left_Dirichlet,
		right_Dirichlet = right_Dirichlet,
		kwargs...)

	return S
end

function D!(S, N; stencil_gridpts = Dict(6=>3, 42=>5), #= Dictionary of gridpt => order of derivative upto that gridpt =#
	left_edge_npts = 2, left_edge_ghost = false,
	right_edge_npts = 2, right_edge_ghost = false,
	left_Dirichlet = false,
	right_Dirichlet = false,
	kwargs...)

	@assert(N≥2,"Need at least 2 points to compute the derivative")

	N_cols = N
	if left_edge_ghost
		N_cols += 1
	end
	if right_edge_ghost
		N_cols += 1
	end

	@assert size(S) == (N, N_cols) "Size of matrix must be $((N, N_cols))"

	gridpt_cutoffs_list = sort(collect(keys(stencil_gridpts)))
	maxorder = maximum(values(stencil_gridpts))

	S .= 0

	# Derivatives on the boundary

	if left_edge_ghost
		startpt = 2 - div(left_edge_npts, 2)
		endpt = startpt + left_edge_npts-1
		if startpt >=1
			v = @view S[1, startpt:endpt]
			derivStencil!(v, 1, left_edge_npts; kwargs...)
		else
			v = @view S[1, 1:left_edge_npts]
			derivStencil!(v, 1, 1, left_edge_npts-2; kwargs...)
		end
	else
		if left_Dirichlet
			S[1, 1:left_edge_npts-1] .= @view derivStencil(1, 1, left_edge_npts-2; kwargs...)[2:end]
		else
			v = @view S[1, 1:left_edge_npts]
			derivStencil!(v, 1, 0, left_edge_npts-1; kwargs...)
		end
	end

	if right_edge_ghost
		endpt = N +  div(right_edge_npts, 2) + ( left_edge_ghost ? 1 : 0 )
		startpt = endpt - right_edge_npts + 1
		if endpt<=N_cols
			v = @view S[end, startpt:endpt]
			derivStencil!(v, 1, right_edge_npts; kwargs...)
		else
			v = @view S[end, (N_cols-right_edge_npts + 1):N_cols]
			derivStencil!(v, 1, right_edge_npts-2, 1; kwargs...)
		end
	else
		if right_Dirichlet
			S[end, (N_cols-right_edge_npts + 2):N_cols] .= @view derivStencil(1, right_edge_npts-2, 1; kwargs...)[1:end-1]
		else
			v = @view S[end, (N_cols-right_edge_npts + 1):N_cols]
			derivStencil!(v, 1, right_edge_npts-1, 0; kwargs...)
		end
	end

	for gridpt in 2:N-1

		gridpt_cutoff =  ceilsearch(gridpt_cutoffs_list, min(gridpt, N-gridpt + 1))
		if haskey(stencil_gridpts, gridpt_cutoff)
			npts = stencil_gridpts[gridpt_cutoff]
		else
			npts = nextodd(maxorder)
		end

		diagpt = gridpt + ( left_edge_ghost ? 1 : 0 )
		startpt = max(1, diagpt - div(npts, 2)-(diagpt + div(npts, 2) > N_cols ? diagpt + div(npts, 2) - N_cols : 0 ))
		endpt = min(N_cols, startpt + npts -1)
		npts_left = diagpt - startpt
		npts_right = endpt - diagpt

		v = @view S[gridpt, startpt:endpt]
		derivStencil!(v, 1, npts_left, npts_right; kwargs...)
	end

	return S
end

function D²(N; stencil_gridpts = Dict(6=>3, 42=>5),
	left_edge_npts = 3, left_edge_ghost = false, right_edge_npts = 3, right_edge_ghost = false, kwargs...)

	@assert(N≥3,"Need at least 3 points to compute the second derivative")

	N_cols = N
	if left_edge_ghost
		N_cols += 1
	end
	if right_edge_ghost
		N_cols += 1
	end

	S = zeros(N, N_cols)

	gridpt_cutoffs_list = sort(collect(keys(stencil_gridpts)))
	maxorder = maximum(values(stencil_gridpts))

	# Derivatives on the boundary

	if left_edge_ghost
		startpt = 2 - div(left_edge_npts, 2)
		endpt = startpt + left_edge_npts-1
		if startpt >=1
			v = @view S[1, startpt:endpt]
			derivStencil!(v, 2, left_edge_npts; kwargs...)
		else
			v = @view S[1, 1:left_edge_npts]
			derivStencil!(v, 2, 1, left_edge_npts-2; kwargs...)
		end
	else
		v = @view S[1, 1:left_edge_npts]
		derivStencil!(v, 2, 0, left_edge_npts-1; kwargs...)
	end

	if right_edge_ghost
		endpt = N +  div(right_edge_npts, 2) + ( left_edge_ghost ? 1 : 0 )
		startpt = endpt - right_edge_npts + 1
		if endpt<=N_cols
			v = @view S[end, startpt:endpt]
			derivStencil!(v, 2, right_edge_npts; kwargs...)
		else
			v = @view S[end, N_cols-right_edge_npts + 1:N_cols]
			derivStencil!(v, 2, right_edge_npts-2, 1; kwargs...)
		end
	else
		v = @view S[end, N_cols-right_edge_npts + 1:N_cols]
		derivStencil!(v, 2, right_edge_npts-1, 0; kwargs...)
	end

	for gridpt in 2:N-1

		gridpt_cutoff =  ceilsearch(gridpt_cutoffs_list, min(gridpt, N-gridpt + 1))
		if haskey(stencil_gridpts, gridpt_cutoff)
			npts = stencil_gridpts[gridpt_cutoff]
		else
			npts = nextodd(maxorder)
		end

		diagpt = gridpt + ( left_edge_ghost ? 1 : 0 )
		startpt = max(1, diagpt - div(npts, 2)-(diagpt + div(npts, 2) > N_cols ? diagpt + div(npts, 2) - N_cols : 0 ))
		endpt = min(N_cols, startpt + npts -1)
		npts_left = diagpt - startpt
		npts_right = endpt - diagpt

		v = @view S[gridpt, startpt:endpt]
		derivStencil!(v, 2, npts_left, npts_right; kwargs...)
	end

	return S
end

function Dⁿ(order, N; stencil_gridpts = Dict(6=>order + 1, 42=>order + 3),
	left_edge_npts = order + 1, left_edge_ghost = false, right_edge_npts = order + 1, right_edge_ghost = false, kwargs...)

	@assert(N≥order + 1,"Need at least $(order + 1) points to compute the derivative")

	N_cols = N
	if left_edge_ghost
		N_cols += 1
	end
	if right_edge_ghost
		N_cols += 1
	end

	S = zeros(N, N_cols)

	gridpt_cutoffs_list = sort(collect(keys(stencil_gridpts)))
	gridpt_cutoff_maxorder = maximum(keys(stencil_gridpts))
	maxorder = stencil_gridpts[gridpt_cutoff_maxorder]

	# Derivatives on the boundary

	if left_edge_ghost
		startpt = 2 - div(left_edge_npts, 2)
		endpt = startpt + left_edge_npts-1
		if startpt >=1
			v = @view S[1, startpt:endpt]
			derivStencil!(v, order, left_edge_npts; kwargs...)
		else
			v = @view S[1, 1:left_edge_npts]
			derivStencil!(v, order, 1, left_edge_npts-2; kwargs...)
		end
	else
		v = @view S[1, 1:left_edge_npts]
		derivStencil!(v, order, 0, left_edge_npts-1; kwargs...)
	end

	if right_edge_ghost
		endpt = N +  div(right_edge_npts, 2) + ( left_edge_ghost ? 1 : 0 )
		startpt = endpt - right_edge_npts + 1
		if endpt<=N_cols
			v = @view S[end, startpt:endpt]
			derivStencil!(v, order, right_edge_npts; kwargs...)
		else
			v = @view S[end, (N_cols-right_edge_npts + 1):N_cols]
			derivStencil!(v, order, right_edge_npts-2, 1; kwargs...)
		end
	else
		v = @view S[end, (N_cols-right_edge_npts + 1):N_cols]
		derivStencil!(v, order, right_edge_npts-1, 0; kwargs...)
	end

	for gridpt in 2:N-1

		gridpt_cutoff =  ceilsearch(gridpt_cutoffs_list, min(gridpt, N-gridpt + 1))
		if haskey(stencil_gridpts, gridpt_cutoff)
			npts = stencil_gridpts[gridpt_cutoff]
		else
			npts = nextodd(maxorder)
		end

		diagpt = gridpt + ( left_edge_ghost ? 1 : 0 )
		startpt = max(1, diagpt - div(npts, 2)-(diagpt + div(npts, 2) > N_cols ? diagpt + div(npts, 2) - N_cols : 0 ))
		endpt = min(N_cols, startpt + npts -1)
		npts_left = diagpt - startpt
		npts_right = endpt - diagpt

		v = @view S[gridpt, startpt:endpt]
		derivStencil!(v, order, npts_left, npts_right; kwargs...)
	end

	return S
end

function dbydr(dr::AbstractArray; stencil_gridpts = Dict(6=>3, 42=>5), kwargs...)
	D(length(dr); stencil_gridpts = stencil_gridpts, kwargs...) ./ dr
end

function dbydr!(S::AbstractMatrix, dr::AbstractArray; stencil_gridpts = Dict(6=>3, 42=>5), kwargs...)
	D!(S, length(dr); stencil_gridpts = stencil_gridpts, kwargs...)
	S ./= dr
	return S
end

function dbydr(N::Integer, dx::Real, stencil_gridpts = Dict(6=>3, 42=>5), kwargs...)
	D(N; stencil_gridpts = stencil_gridpts, gridspacing = dx)
end

function dbydr!(S::AbstractMatrix, N::Integer, dx::Real, stencil_gridpts = Dict(6=>3, 42=>5); kwargs...)
	D!(S, N; stencil_gridpts = stencil_gridpts, gridspacing = dx)
end

function roll(arr, shift)
	# translation with wrap
	# A′ = T(shift) A
	# translation: A′(i) = A(i-shift)
	# translation with wrap: A′(i + 1) = A(mod(i-shift, N)+1), i = 0...N-1

	N = size(arr, ndims(arr))
	shift = mod(shift, N)
	if shift==0
		return arr
	end

	newarr = similar(arr)

	for i in 0:N-1
		newarr[:,:, i + 1] .= @view arr[:,:, mod(i-shift, N)+1]
	end
	return newarr
end
