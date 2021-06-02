function fft_t_to_ω(arr::AbstractArray,dt,dim=1) 
	rfft(arr,dim) .* dt
end
function fft_t_to_ω(arr::OffsetArray,dt,dim=1)
	rfft(parent(arr),dim) .* dt
end

function fft_ω_to_t(arr::AbstractArray,dν,dim=1,Nt=2*(size(arr,dim)-1))
	brfft(arr,Nt,dim) .* dν
end
function fft_ω_to_t(arr::OffsetArray,dν,dim=1,Nt=2*(size(arr,dim)-1))
	brfft(parent(arr),Nt,dim) .* dν
end
