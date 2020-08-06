module Timer_utils
	
using Reexport

@reexport using TimerOutputs
@reexport using ProgressMeter
@reexport using ParallelUtilities

using NamedArrays

export print_timer
export pmapbatch_timed
export pmapsum_timed
export pmapreduce_timed
export progress_bar_and_timer
export progress_channel_and_bar
export signaltomaster!

########################################################################################
# Print average and max timings
########################################################################################

function print_timer(all_timers::Vector{TimerOutput})

	if isempty(all_timers)
		println("No timers received from workers")
		return
	end

	N = length(all_timers)

	K = collect(keys(first(all_timers).inner_timers))

	if isempty(K)
		println("No timings recorded on workers")
		return
	end

	println("Average runtimes on workers")

	headers = ["ncalls","avg time","max time","time/call",
	"avg alloc","max alloc","alloc/call"]
	n = NamedArray{Float64}(length(K),length(headers))
	n_pretty = NamedArray{String}(length(K),length(headers))
	setnames!(n,K,1)
	setnames!(n,headers,2)
	setnames!(n_pretty,headers,2)

	for k in K

		ncalls = maximum(t[k].accumulated_data.ncalls for t in all_timers)
		timer_sum = sum(t[k].accumulated_data for t in all_timers)
		max_time = maximum(t[k].accumulated_data.time for t in all_timers)
		max_alloc = maximum(t[k].accumulated_data.allocs for t in all_timers)
		avg_time = timer_sum.time/N
		time_call = avg_time/ncalls
		avg_alloc = timer_sum.allocs/N
		alloc_call = avg_alloc/ncalls

		n[k,:] = [ncalls,avg_time,max_time,time_call,avg_alloc,max_alloc,alloc_call]
	end

	p = sortperm(@view(n[:,"avg time"]),rev=true)
	n = n[p,:]

	setnames!(n_pretty,names(n,1),1)

	for k in K
		n_pretty[k,"ncalls"] = string(Int(n[k,"ncalls"]))
		n_pretty[k,"avg time"] = TimerOutputs.prettytime(n[k,"avg time"])
		n_pretty[k,"max time"] = TimerOutputs.prettytime(n[k,"max time"])
		n_pretty[k,"time/call"] = TimerOutputs.prettytime(n[k,"time/call"])
		n_pretty[k,"avg alloc"] = TimerOutputs.prettymemory(n[k,"avg alloc"])
		n_pretty[k,"max alloc"] = TimerOutputs.prettymemory(n[k,"max alloc"])
		n_pretty[k,"alloc/call"] = TimerOutputs.prettymemory(n[k,"alloc/call"])
	end

	display(n_pretty) 
end

function print_timer(timers_channel::RemoteChannel,np::Int)
	print_timer([take!(timers_channel) for i=1:np])
end

function print_timer(timers_channel::RemoteChannel,np::Int,total::TimerOutput)
	print_timer(timers_channel,np)
	println("Total time")
	println(total)
	if haskey(total.inner_timers,"pmapsum")
		pmapsumtime = total.inner_timers["pmapsum"].accumulated_data.time
		println("Total time on $np processors : $(TimerOutputs.prettytime(np*pmapsumtime))")
	elseif haskey(total.inner_timers,"pmap")
		pmapsumtime = total.inner_timers["pmap"].accumulated_data.time
		println("Total time on $np processors : $(TimerOutputs.prettytime(np*pmapsumtime))")
	end
end

print_timer(timers_channel::RemoteChannel,N::Integer,::Nothing) = 
	print_timer(timers_channel,N)

########################################################################################
# timed pmapsum and pmapreduce
########################################################################################

function pmapbatch_timed(f::Function,
	iterproduct::Iterators.ProductIterator,args...;
	progress_str::String="Modes summed : ",
	print_timings::Bool=false,
	nprogressticks=length(iterproduct),
	kwargs...)

	progress_channel,progress_bar,timers_channel,total_time,np = 
		progress_bar_and_timer(iterproduct,progress_str,nprogressticks)

	# f_padded(ps) = f(ps,args...,nothing,nothing)
	f_padded(ps) = f(ps,args...,progress_channel,timers_channel)

	# These are useful for debugging
	# ps = ProductSplit(iterproduct.iterators,np,1)
	# res = f_padded(ps)
	# res = pmapsum(f_padded,iterproduct)

	res = nothing
	
	t = @async begin
		while take!(progress_channel)
			next!(progress_bar)
		end
		finish!(progress_bar)
		nothing
	end
	
	try
		@timeit total_time "pmap" begin
			res = pmapbatch(f_padded,iterproduct.iterators)
		end
	finally
		put!(progress_channel,false)
		finalize(progress_channel)
	end

	print_timings && print_timer(timers_channel,np,total_time)
	finalize(timers_channel)
	
	wait(t)
	res
end

function pmapsum_timed(f::Function,
	iterproduct::Iterators.ProductIterator,args...;
	progress_str::String="Modes summed : ",
	print_timings::Bool=false,
	nprogressticks=length(iterproduct),
	showprogress = false,
	kwargs...)

	progress_channel,progress_bar,timers_channel,total_time,np = 
		progress_bar_and_timer(iterproduct,progress_str,nprogressticks)

	# f_padded(ps) = f(ps,args...,nothing,nothing)
	f_padded(ps) = f(ps,args...,progress_channel,timers_channel)

	# These are useful for debugging
	# ps = ProductSplit(iterproduct.iterators,np,1)
	# res = f_padded(ps)
	# res = pmapsum(f_padded,iterproduct.iterators)

	res = nothing

	@sync begin
		@async try
			@timeit total_time "pmapsum" begin
				res = pmapsum(f_padded,iterproduct.iterators,
						showprogress=showprogress)
			end
		finally
			put!(progress_channel,false)
			finalize(progress_channel)
		end

		while take!(progress_channel)
			next!(progress_bar)
		end
		finish!(progress_bar)
	end

	print_timings && print_timer(timers_channel,np,total_time)
	finalize(timers_channel)
	
	res
end

function pmapreduce_timed(fmap::Function,
	freduce::Function,
	iterproduct::Iterators.ProductIterator,args...;
	progress_str::String="Modes summed : ",
	print_timings::Bool=false,
	nprogressticks=length(iterproduct),
	showprogress = false,
	kwargs...)

	progress_channel,progress_bar,timers_channel,total_time,np = 
		progress_bar_and_timer(iterproduct,progress_str,nprogressticks)

	fmap_padded(ps) = fmap(ps,args...,progress_channel,timers_channel)
	
	res = nothing

	# t = @async begin
	# 	while take!(progress_channel)
	# 		next!(progress_bar)
	# 	end
	# 	finish!(progress_bar)
	# 	nothing
	# end

	# res = pmapreduce(fmap_padded,freduce,iterproduct)

	@sync begin
		@async try
			@timeit total_time "pmapsum" begin
				res = pmapreduce(fmap_padded,freduce,iterproduct.iterators,
						showprogress=showprogress)
			end
		catch
			rethrow()
		finally
			put!(progress_channel,false)
			finalize(progress_channel)
		end

		try
			while take!(progress_channel)
				next!(progress_bar)
			end
		finally
			finish!(progress_bar)
		end
	end

	print_timings && print_timer(timers_channel,np,total_time)
	finalize(timers_channel)

	res
end

for pmapfn in [:pmapsum_timed,:pmapbatch_timed,:pmapreduce_timed]
	@eval function $pmapfn(f,r::AbstractRange,args...;kwargs...)
		$pmapfn(f,Iterators.product(r),args...;kwargs...)
	end
	@eval function $pmapfn(f,t::Tuple{Vararg{AbstractRange}},args...;kwargs...)
		$pmapfn(f,Iterators.product(t...),args...;kwargs...)
	end
end

function progress_bar_and_timer(iterproduct,progress_str="Modes summed : ",nprogressticks=length(iterproduct))
	np = ParallelUtilities.nworkersactive(iterproduct.iterators)
	progress_bar = Progress(nprogressticks,1,progress_str)

	# Extra element for `false` tag at the end
	modes_progress_channel = RemoteChannel(()->Channel{Bool}(nprogressticks+1))

	timers_channel = RemoteChannel(()->Channel{TimerOutput}(np))

	modes_progress_channel,progress_bar,timers_channel,TimerOutput(),np
end

function progress_channel_and_bar(iterproduct,progress_str="Modes summed : ",nprogressticks=length(iterproduct))
	progress_bar = Progress(nprogressticks,1,progress_str)

	# Extra element for `false` tag at the end
	modes_progress_channel = RemoteChannel(()->Channel{Bool}(nprogressticks+1))

	modes_progress_channel,progress_bar
end

########################################################################################
# utility functions
########################################################################################

@inline signaltomaster!(r::RemoteChannel{Channel{Bool}}) = put!(r,true)
@inline signaltomaster!(r::RemoteChannel{Channel{TimerOutput}},timer::TimerOutput) = put!(r,timer)
# Ignore anything other than RemoteChannels
@inline signaltomaster!(r) = nothing
@inline signaltomaster!(r,val) = nothing

end