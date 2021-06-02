########################################################################################
# timed pmapsum and pmapreduce
########################################################################################
function pmapsum(comm::MPI.Comm, f, iters, args...; kwargs...)
	print_timings = get(kwargs, :print_timings, false)
	localtimer = TimerOutput()
	ℓ_ωind_iter_on_proc = productsplit(iters, comm)
	res_local = f(localtimer, ℓ_ωind_iter_on_proc, args...; kwargs...)
	if print_timings && res_local !== nothing
		println(localtimer)
	end
	MPI.Reduce(res_local, MPI.SUM, 0, comm)
end

function pmapsum(::Any, f, iters, args...; kwargs...)
	print_timings = get(kwargs, :print_timings, false)
	fwrapper(ps) = begin
		localtimer = TimerOutput()
		res = f(localtimer, ps, args...)
		if print_timings && ParallelUtilities.workerrank(ps) == 1
			println(localtimer)
		end
		res
	end

	pmapreduce_productsplit(fwrapper, ParallelUtilities.elementwisesum!, iters...)
end

function pmapreduce(::Any, f, fred, iters, args...;	kwargs...)
	print_timings = get(kwargs, :print_timings, false)
	fwrapper(ps) = begin
		localtimer = TimerOutput()
		res = f(localtimer, ps, args...)
		if print_timings && ParallelUtilities.workerrank(ps) == 1
			println(localtimer)
		end
		res
	end

	pmapreduce_productsplit(fwrapper, fred, iters...)
end

########################################################################################
# utility functions
########################################################################################

signaltomaster!(r::RemoteChannel{Channel{Bool}}) = put!(r, true)
signaltomaster!(r::RemoteChannel{Channel{TimerOutput}}, timer::TimerOutput) = put!(r, timer)
# Ignore anything other than RemoteChannels
signaltomaster!(val...) = nothing

_broadcast(A, root, comm) = A
_broadcast(A, root, comm::MPI.Comm) = MPI.bcast(A, root, comm)
