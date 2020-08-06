###########################################################################################
# Operators
###########################################################################################

module Finite_difference

	using Reexport
	@reexport using SparseArrays

	function derivStencil(order::Integer,ptsleft::Integer,ptsright::Integer;gridspacing::Real=1)
		
		M = zeros(ptsleft + ptsright+1,ptsleft + ptsright+1)

		for (p_ind,p) in enumerate(0:(ptsleft + ptsright)) , (m_ind,m) in enumerate(-ptsleft:ptsright)
			M[p_ind,m_ind] = m^p/factorial(p)
		end

		M_order = inv(M)[:,order+1]./gridspacing^order
		
		if (ptsleft == ptsright) && isodd(order)
			# Fix loss of numerical precision
			M_order[ptsleft+1] = 0.
		end

		stencil = sparsevec(M_order)
	end

	function derivStencil(order::Integer,pts::Integer;kwargs...) 
		if isodd(pts)
			return derivStencil(order,div(pts,2),div(pts,2);kwargs...)
		else
			return derivStencil(order,div(pts,2)-1,div(pts,2);kwargs...)
		end
	end

	nextodd(n::Integer) = isodd(n) ? n+2 : n+1

	function ceilsearch(arr,n) 
		# assume sorted
		for elem in arr
			if elem>=n
				return elem
			end
		end
		return n
	end

	# Dictionary of gridpt => order of derivative upto that gridpt

	function D(N,stencil_gridpts=Dict(6=>3,42=>5);
		left_edge_npts=2,left_edge_ghost=false,
		right_edge_npts=2,right_edge_ghost=false,
		left_Dirichlet=false,
		right_Dirichlet=false,
		kwargs...)

		@assert(N≥2,"Need at least 2 points to compute the derivative")
		
		N_cols = N
		if left_edge_ghost
			N_cols += 1
		end
		if right_edge_ghost
			N_cols += 1
		end

		S = spzeros(N,N_cols)

		gridpt_cutoffs_list = sort(collect(keys(stencil_gridpts)))
		gridpt_cutoff_maxorder = maximum(keys(stencil_gridpts))
		maxorder = stencil_gridpts[gridpt_cutoff_maxorder]

		# Derivatives on the boundary

		if left_edge_ghost
			startpt = 2 - div(left_edge_npts,2)
			endpt = startpt+left_edge_npts-1
			if startpt >=1 
				S[1,startpt:endpt] = derivStencil(1,left_edge_npts;kwargs...)
			else
				S[1,1:left_edge_npts] = derivStencil(1,1,left_edge_npts-2;kwargs...)
			end
		else
			if left_Dirichlet
				S[1,1:left_edge_npts-1] = derivStencil(1,1,left_edge_npts-2;kwargs...)[2:end]
			else
				S[1,1:left_edge_npts] = derivStencil(1,0,left_edge_npts-1;kwargs...)
			end
		end

		if right_edge_ghost
			endpt = N +  div(right_edge_npts,2) + ( left_edge_ghost ? 1 : 0 )
			startpt = endpt - right_edge_npts + 1
			if endpt<=N_cols
				S[end,startpt:endpt] = derivStencil(1,right_edge_npts;kwargs...)
			else
				S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(1,right_edge_npts-2,1;kwargs...)
			end
		else
			if right_Dirichlet
				S[end,(N_cols-right_edge_npts+2):N_cols] = derivStencil(1,right_edge_npts-2,1;kwargs...)[1:end-1]
			else
				S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(1,right_edge_npts-1,0;kwargs...)
			end
		end

		for gridpt in 2:N-1

			gridpt_cutoff =  ceilsearch(gridpt_cutoffs_list,min(gridpt,N-gridpt+1))
			if haskey(stencil_gridpts,gridpt_cutoff)
				npts = stencil_gridpts[gridpt_cutoff]
			else
				npts = nextodd(maxorder)
			end

			diagpt = gridpt + ( left_edge_ghost ? 1 : 0 )
			startpt = max(1,diagpt - div(npts,2)-(diagpt + div(npts,2) > N_cols ? diagpt + div(npts,2) - N_cols : 0 ))
			endpt = min(N_cols,startpt + npts -1)
			npts_left = diagpt - startpt
			npts_right = endpt - diagpt

			S[gridpt,startpt:endpt] = derivStencil(1,npts_left,npts_right;kwargs...)
		end

		return S
	end

	function D²(N,stencil_gridpts=Dict(6=>3,42=>5);
		left_edge_npts=3,left_edge_ghost=false,right_edge_npts=3,right_edge_ghost=false,kwargs...)

		@assert(N≥3,"Need at least 3 points to compute the second derivative")
		
		N_cols = N
		if left_edge_ghost
			N_cols += 1
		end
		if right_edge_ghost
			N_cols += 1
		end

		S = spzeros(N,N_cols)

		gridpt_cutoffs_list = sort(collect(keys(stencil_gridpts)))
		gridpt_cutoff_maxorder = maximum(keys(stencil_gridpts))
		maxorder = stencil_gridpts[gridpt_cutoff_maxorder]

		# Derivatives on the boundary

		if left_edge_ghost
			startpt = 2 - div(left_edge_npts,2)
			endpt = startpt+left_edge_npts-1
			if startpt >=1 
				S[1,startpt:endpt] = derivStencil(2,left_edge_npts;kwargs...)
			else
				S[1,1:left_edge_npts] = derivStencil(2,1,left_edge_npts-2;kwargs...)
			end
		else
			S[1,1:left_edge_npts] = derivStencil(2,0,left_edge_npts-1;kwargs...)
		end

		if right_edge_ghost
			endpt = N +  div(right_edge_npts,2) + ( left_edge_ghost ? 1 : 0 )
			startpt = endpt - right_edge_npts + 1
			if endpt<=N_cols
				S[end,startpt:endpt] = derivStencil(2,right_edge_npts;kwargs...)
			else
				S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(2,right_edge_npts-2,1;kwargs...)
			end
		else
			S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(2,right_edge_npts-1,0;kwargs...)
		end

		for gridpt in 2:N-1

			gridpt_cutoff =  ceilsearch(gridpt_cutoffs_list,min(gridpt,N-gridpt+1))
			if haskey(stencil_gridpts,gridpt_cutoff)
				npts = stencil_gridpts[gridpt_cutoff]
			else
				npts = nextodd(maxorder)
			end

			diagpt = gridpt + ( left_edge_ghost ? 1 : 0 )
			startpt = max(1,diagpt - div(npts,2)-(diagpt + div(npts,2) > N_cols ? diagpt + div(npts,2) - N_cols : 0 ))
			endpt = min(N_cols,startpt + npts -1)
			npts_left = diagpt - startpt
			npts_right = endpt - diagpt

			S[gridpt,startpt:endpt] = derivStencil(2,npts_left,npts_right;kwargs...)
		end

		return S
	end

	function Dⁿ(order,N,stencil_gridpts=Dict(6=>order+1,42=>order+3);
		left_edge_npts=order+1,left_edge_ghost=false,right_edge_npts=order+1,right_edge_ghost=false,kwargs...)

		@assert(N≥order+1,"Need at least $(order+1) points to compute the derivative")
		
		N_cols = N
		if left_edge_ghost
			N_cols += 1
		end
		if right_edge_ghost
			N_cols += 1
		end

		S = spzeros(N,N_cols)

		gridpt_cutoffs_list = sort(collect(keys(stencil_gridpts)))
		gridpt_cutoff_maxorder = maximum(keys(stencil_gridpts))
		maxorder = stencil_gridpts[gridpt_cutoff_maxorder]

		# Derivatives on the boundary

		if left_edge_ghost
			startpt = 2 - div(left_edge_npts,2)
			endpt = startpt+left_edge_npts-1
			if startpt >=1 
				S[1,startpt:endpt] = derivStencil(order,left_edge_npts;kwargs...)
			else
				S[1,1:left_edge_npts] = derivStencil(order,1,left_edge_npts-2;kwargs...)
			end
		else
			S[1,1:left_edge_npts] = derivStencil(order,0,left_edge_npts-1;kwargs...)
		end

		if right_edge_ghost
			endpt = N +  div(right_edge_npts,2) + ( left_edge_ghost ? 1 : 0 )
			startpt = endpt - right_edge_npts + 1
			if endpt<=N_cols
				S[end,startpt:endpt] = derivStencil(order,right_edge_npts;kwargs...)
			else
				S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(order,right_edge_npts-2,1;kwargs...)
			end
		else
			S[end,(N_cols-right_edge_npts+1):N_cols] = derivStencil(order,right_edge_npts-1,0;kwargs...)
		end

		for gridpt in 2:N-1

			gridpt_cutoff =  ceilsearch(gridpt_cutoffs_list,min(gridpt,N-gridpt+1))
			if haskey(stencil_gridpts,gridpt_cutoff)
				npts = stencil_gridpts[gridpt_cutoff]
			else
				npts = nextodd(maxorder)
			end

			diagpt = gridpt + ( left_edge_ghost ? 1 : 0 )
			startpt = max(1,diagpt - div(npts,2)-(diagpt + div(npts,2) > N_cols ? diagpt + div(npts,2) - N_cols : 0 ))
			endpt = min(N_cols,startpt + npts -1)
			npts_left = diagpt - startpt
			npts_right = endpt - diagpt

			S[gridpt,startpt:endpt] = derivStencil(order,npts_left,npts_right;kwargs...)
		end

		return S
	end

	function dbydr(dr::AbstractArray,stencil_gridpts=Dict(6=>3,42=>5);kwargs...)
		dropzeros(D(length(dr),stencil_gridpts;kwargs...) ./ dr)
	end

	function dbydr(N::Integer,dx::Real,stencil_gridpts=Dict(6=>3,42=>5);kwargs...)
		dropzeros(D(N,stencil_gridpts;kwargs...) ./ dx)
	end

	function roll(arr,shift)
		# translation with wrap
		# A′ = T(shift) A
		# translation: A′(i) = A(i-shift)
		# translation with wrap: A′(i+1) = A(mod(i-shift,N)+1), i=0...N-1

		N = size(arr,ndims(arr))
		shift = mod(shift,N)
		if shift==0
			return arr
		end

		newarr = similar(arr)
		
		for i in 0:N-1
			newarr[:,:,i+1] .= arr[:,:,mod(i-shift,N)+1]
		end
		return newarr
	end

	export D,D²,Dⁿ,dbydr,derivStencil,roll
end