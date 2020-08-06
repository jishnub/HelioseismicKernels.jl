module Traveltimes

using ..Kernel

los_tag_string(::Directions.los_radial) = ""
los_tag_string(::Directions.los_earth) = "_los"

# Flows

uniform_rotation_u⁺(Ω_rot=20e2/Rsun) = @. √(4π/3)*im*Ω_rot*r

function δτ_uniform_rotation_int_K_u(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)
	
	K₊ = kernel_ℑu⁺₁₀(TravelTimes(),xobs1,xobs2,los;kwargs...)
	
	Ω_rot=get(kwargs,:Ω_rot,20e2/Rsun)
	u⁺ = uniform_rotation_u⁺(Ω_rot)

	δτ_v_FB = simps((@.r^2 * K₊*imag(u⁺)),r)
	@save_to_fits_and_return(δτ_v_FB)
end

function δτ_uniform_rotation_int_hω_δCω_FB(n1,n2,
	los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dω,ν_start_zeros,ν_end_zeros,Nν_Gfn = p_Gsrc

	h_ω = get(kwargs,:hω) do 
		hω(TravelTimes(),n1,n2,los;kwargs...)
	end

	δCω = δCω_uniform_rotation_firstborn_integrated_over_angle(n1,n2,los;kwargs...)
	δCω = δCω[axes(h_ω,1) .+ ν_start_zeros]

	δτ = sum(@. 2real(conj(h_ω)*δCω))*dω/2π
end

function δτ_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1,n2,
	los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dω,ν_start_zeros,ν_end_zeros,Nν_Gfn = p_Gsrc

	h_ω = get(kwargs,:hω) do 
		hω(TravelTimes(),n1,n2,los;kwargs...)
	end

	δCω = δCω_uniform_rotation_rotatedwaves_linearapprox(n1,n2,los;kwargs...)
	δCω = δCω[axes(h_ω,1) .+ ν_start_zeros]

	δτ = sum(@. 2real(conj(h_ω)*δCω))*dω/2π
end

function δτ_uniform_rotation_rotatedframe_int_hω_δCω(n1,n2,
	los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dω,ν_start_zeros,ν_end_zeros,Nν_Gfn = p_Gsrc

	h_ω = get(kwargs,:hω) do 
		hω(TravelTimes(),n1,n2,los;kwargs...)
	end

	δCω = δCω_uniform_rotation_rotatedwaves(n1,n2,los;kwargs...)
	δCω_in_range = δCω[axes(h_ω,1) .+ ν_start_zeros]

	δτ = sum(@. 2real(conj(hω)*δCω_in_range))*dω/2π
	δτ,δCω
end

function δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1,n2,
	los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dt,dν,Nt = p_Gsrc

	Cω_n1n2 = get(kwargs,:Cω) do 
		Cω(n1,n2,los;kwargs...)
	end
	Ct_n1n2 = get(kwargs,:Ct) do 
		Ct(Cω_n1n2,los;kwargs...,dν=dν)
	end

	τ_ind_range = get(kwargs,:τ_ind_range) do 
		time_window_indices_by_fitting_bounce_peak(Ct_n1n2,n1,n2;
			dt=dt,Nt=Nt,kwargs...)
	end

	h_t = get(kwargs,:ht) do 
		ht(TravelTimes(),Cω_n1n2,n1,n2;τ_ind_range=τ_ind_range,dν=dν,kwargs...)
	end
	h_t = @view h_t[τ_ind_range]

	δCt = δCt_uniform_rotation_rotatedwaves(n1,n2,los;
			τ_ind_range=τ_ind_range,Cω=Cω_n1n2,dν=dν,kwargs...)

	δτ = simps(h_t.*parent(δCt),dt)
	δτ,δCt
end

function δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1,n2,
	los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dt,dν,Nt = p_Gsrc

	if isnothing(get(kwargs,:Cω,nothing)) && isnothing(get(kwargs,:∂ϕ₂Cω,nothing))
		Cω_n1n2,∂ϕ₂Cω_n1n2 = Cω_∂ϕ₂Cω(n1,n2,los;kwargs...)
	else
		Cω_n1n2 = get(kwargs,:Cω) do 
					Cω(n1,n2,los;kwargs...)
			end
		∂ϕ₂Cω_n1n2 = get(kwargs,:∂ϕ₂Cω) do 
				∂ϕ₂Cω(n1,n2,los;kwargs...)
			end
	end

	τ_ind_range = get(kwargs,:τ_ind_range) do 
		time_window_indices_by_fitting_bounce_peak(Ct(Cω_n1n2,los,dν=dν),
							n1,n2;dt=dt,Nt=Nt,kwargs...)
	end

	h_t = get(kwargs,:ht) do
		h_ω = get(kwargs,:hω,nothing)
		!isnothing(h_ω) ? fft_ω_to_t(h_ω,dν) :
		ht(TravelTimes(),Cω_n1n2,n1,n2;τ_ind_range=τ_ind_range,dν=dν,kwargs...)
	end
	h_t = h_t[τ_ind_range]

	δCt = δCt_uniform_rotation_rotatedwaves_linearapprox(
			∂ϕ₂Ct(∂ϕ₂Cω_n1n2,dν=dν);kwargs...,τ_ind_range=τ_ind_range)
	
	δτ = simps((@. h_t*δCt),dt)
	δτ,δCt
end

function δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(
	n1::Point2D,n2_arr::Vector{<:Point2D},
	los::los_direction=los_radial();kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	
	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dt,dν,Nt = p_Gsrc

	Cω_n1n2 = get(kwargs,:Cω) do 
				Cω(n1,n2_arr,los;kwargs...)
			end

	∂ϕ₂Cω_n1n2 = get(kwargs,:∂ϕ₂Cω) do 
					∂ϕ₂Cω(n1,n2_arr,los;kwargs...)
				end
	
	τ_ind_range = get(kwargs,:τ_ind_range) do
					time_window_indices_by_fitting_bounce_peak(
					Ct(Cω_n1n2,los,dν=dν),n1,n2_arr;dt=dt,Nt=Nt,kwargs...)
	end

	h_ω = get(kwargs,:hω,nothing)
	h_t = get(kwargs,:ht) do 
		!isnothing(h_ω) ? fft_ω_to_t(h_ω,dν) :
		ht(TravelTimes(),Cω_n1n2,n1,n2;τ_ind_range=τ_ind_range,dν=dν,kwargs...)
	end

	∂ϕ₂Ct_n1n2 = ∂ϕ₂Ct(∂ϕ₂Cω_n1n2,dν=dν)
	δCt = δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_n1n2;kwargs...)

	δτ = zeros(length(n2_arr))
	for (n2ind,τ_inds) in enumerate(τ_ind_range)
		δτ[n2ind] = simps((@. h_t[τ_inds,n2ind]*δCt[τ_inds,n2ind]),dt)
	end
	δτ,δCt
end

function validate(::flows,n1,n2,los::los_direction=los_radial();kwargs...)

	dt,dν = read_parameters("dt","dν";kwargs...)

	C_ω,∂ϕ₂_Cω = Cω_∂ϕ₂Cω(n1,n2,los;kwargs...)
	h_t = ht(TravelTimes(),C_ω,n1,n2;kwargs...)
	h_ω = hω(TravelTimes(),C_ω,n1,n2;kwargs...)

	s_arr = String[]

	δτ, = δτ_uniform_rotation_int_K_u(n1,n2,los;hω=h_ω,kwargs...)
	s = @sprintf "%-50s %g" "First Born, ∫dr u(r) K(r)" round(δτ,sigdigits=3)
	push!(s_arr,s)

	δτ, = δτ_uniform_rotation_int_hω_δCω_FB(n1,n2,los;hω=h_ω,kwargs...)
	s = @sprintf "%-50s %g" "First Born, ∫dω/2π h(ω) δC_FB(ω)" round(δτ,sigdigits=3)
	push!(s_arr,s)

	δτ, = δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1,n2,los;ht=h_t,Cω=C_ω,kwargs...)
	s = @sprintf "%-50s %g" "Rotated frame, ∫dt h(t) δC_R(t)" round(δτ,sigdigits=3)
	push!(s_arr,s)

	δτ, = δτ_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1,n2,los;
					hω=h_ω,Cω=C_ω,∂ϕ₂Cω=∂ϕ₂_Cω,kwargs...)
	s = @sprintf "%-50s %g" "Rotated frame, ∫dω/2π h(ω) δC_R_lin(ω)" round(δτ,sigdigits=3)
	push!(s_arr,s)

	δτ, = δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1,n2,los;ht=h_t,
				Cω=C_ω,∂ϕ₂Cω=∂ϕ₂_Cω,kwargs...)
	s = @sprintf "%-50s %g" "Rotated frame, ∫dt h(t) δC_R_lin(t)" round(δτ,sigdigits=3)
	push!(s_arr,s)

	println("\nTraveltimes\n")
	for s in s_arr
		println(s)
	end
end

function δτ_Δϕ(::flows,los::los_radial=los_radial();kwargs...)

	@unpack Nt,dt,dν,Nν,dω,ν_start_zeros,Nν_Gfn = read_all_parameters(;kwargs...)

	# ϕ in degrees
	ϕ_low = get(kwargs,:ϕ_low,35)
	ϕ_high = get(kwargs,:ϕ_high,65)
	nϕ=get(kwargs,:nϕ,2)
	ϕ2_deg = LinRange(ϕ_low,ϕ_high,nϕ)
	ϕ2_arr = deg2rad.(ϕ2_deg)
	n1 = Point2D(Equator(),0)
	n2_arr = [Point2D(Equator(),ϕ2) for ϕ2 in ϕ2_arr]

	τ_ind_ranges = Vector{UnitRange}(undef,length(n2_arr))

	bounce_no = get(kwargs,:bounce_no,1)

	δτ_v_CmC0 = zeros(length(n2_arr))

	C_ω = Cω(n1,n2_arr,los;kwargs...)
	C_t = Ct(C_ω,dν=dν)
	hω_arr = zeros(ComplexF64,1:Nν_Gfn,length(n2_arr))
	
	@showprogress 1 "Computing δτ_rot : " for (n2ind,n2) in enumerate(n2_arr)

		τ_ind_ranges[n2ind] = time_window_indices_by_fitting_bounce_peak(
								@view(C_t[:,n2ind]),n1,n2,dt=dt)

		h_t = ht(TravelTimes(),@view(C_ω[:,n2ind]),n1,n2;kwargs...,
							τ_ind_range = τ_ind_ranges[n2ind])
		h_ω = fft_t_to_ω(h_t,dt)
		@. hω_arr[:,n2ind] = h_ω[ν_start_zeros .+ (1:Nν_Gfn)]
		
		δτ_v_CmC0[n2ind] = δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1,n2,los;
							Cω = @view(C_ω[:,n2ind]),ht=h_t,
							τ_ind_range = τ_ind_ranges[n2ind],kwargs...)[1]
	end

	δτ_v_FB = δτ_uniform_rotation_int_K_u(n1,n2_arr,los;
				hω=hω_arr,kwargs...)

	header = FITSHeader(["phi_low","phi_high"],[ϕ_low,ϕ_high],
		["Minimum azimuth","Maximum azimuth"])

	los_tag = los_tag_string(los)
	filename = joinpath(SCRATCH_kerneldir,"δτ_v_CmC0"*los_tag*".fits")
	FITS(filename,"w") do f
		write(f,δτ_v_CmC0,header=header)
	end

	filename = joinpath(SCRATCH_kerneldir,"δτ_v_FB"*los_tag*".fits")
	FITS(filename,"w") do f
		write(f,δτ_v_FB,header=header)
	end

	percentage_diff = @. (δτ_v_FB-δτ_v_CmC0)/δτ_v_CmC0*100
	display(hcat(collect(ϕ2_deg),δτ_v_FB,δτ_v_CmC0,percentage_diff))

	δτ_v_FB,δτ_v_CmC0
end

# Sound speed

# Y₀₀ = 1/√(4π), so δc₀₀(r) = √(4π) δc(r)
δc₀₀(δc_scale=1e-5) = @. δc_scale*c * √(4π)

function δτ_isotropic_δc_int_K_δc(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	K = kernel_δc₀₀(TravelTimes(),xobs1,xobs2,los;kwargs...)
	c_scale = get(kwargs,:c_scale,1+1e-5)
	δc = δc₀₀(c_scale-1)

	δτ_δc_FB = simps((@. r^2 *K* δc),r)
	# @save_to_fits_and_return(δτ_δc_FB,los)
end

function δτ_isotropic_δc_int_hω_δCω_FB(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dω,ν_start_zeros,ν_end_zeros,Nν_Gfn = p_Gsrc

	h_ω = get(kwargs,:hω) do 
		hω(TravelTimes(),xobs1,xobs2,los;kwargs...,c_scale=1)
	end

	c_scale = get(kwargs,:c_scale,1+1e-5)

	δCω = δCω_isotropicδc_firstborn_integrated_over_angle(xobs1,xobs2,los;kwargs...,
		c_scale=c_scale)
	δCω = δCω[axes(h_ω,1) .+ ν_start_zeros]

	δτ = sum(@. 2real(conj(h_ω)*δCω))*dω/2π
end

function δτ_isotropic_δc_int_ht_δCt_CmC0(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)
	
	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dt,Nt,dν = p_Gsrc
	
	c_scale = get(kwargs,:c_scale,1+1e-5)
	bounce_no = get(kwargs,:bounce_no,1)

	Cω_xobs1xobs2 = get(kwargs,:Cω,Cω(xobs1,xobs2,los;kwargs...,c_scale=1))
	Ct_xobs1xobs2 = get(kwargs,:Ct,Ct(Cω_xobs1xobs2,dν=dν))
	τ_ind_range = time_window_indices_by_fitting_bounce_peak(Ct_xobs1xobs2,xobs1,xobs2,
					dt=dt,Nt=Nt,bounce_no=bounce_no)

	h_t = get(kwargs,:ht) do
		ht(TravelTimes(),Cω_xobs1xobs2,xobs1,xobs2;
			τ_ind_range=τ_ind_range,kwargs...)
	end

	δCω = get(kwargs,:δCω,nothing)
	if isnothing(δCω)
		C′t_xobs1xobs2 = Ct(xobs1,xobs2,los;kwargs...,c_scale=c_scale)
		δCt = @. C′t_xobs1xobs2 - Ct_xobs1xobs2
	else
		δCt = δCt_isotropicδc_C_minus_C0(δCω;kwargs...)
	end

	δτ_δc_CmC0 = simps((@. h_t *δCt),dt)
	# @save_to_fits_and_return(δτ_δc_CmC0,los)
	δτ_δc_CmC0
end

function δτ_isotropic_δc_int_hω_δCω_CmC0(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dω,ν_start_zeros,ν_end_zeros,Nν_Gfn = p_Gsrc

	h_ω = get(kwargs,:hω) do 
		hω(TravelTimes(),xobs1,xobs2,los;kwargs...,c_scale=1)
	end

	c_scale = get(kwargs,:c_scale,1+1e-5)

	δCω = δCω_isotropicδc_C_minus_C0(xobs1,xobs2,los;c_scale=c_scale,kwargs...)
	δCω_in_range = δCω[axes(h_ω,1) .+ ν_start_zeros,..] |> parent

	h_ω = parent(h_ω)

	δτ = sum(@. 2real(conj(h_ω)*δCω_in_range))*dω/2π
	δτ,δCω
end

function validate(::soundspeed,n1,n2,los::los_direction=los_radial();kwargs...)

	p_Gsrc = read_all_parameters(;kwargs...)
	@unpack dω,ν_start_zeros,ν_end_zeros,Nν_Gfn,dt,dν = p_Gsrc

	C_ω = Cω(n1,n2,los;kwargs...,c_scale=1)
	h_t = ht(TravelTimes(),C_ω,n1,n2;kwargs...)
	h_ω = hω(TravelTimes(),C_ω,n1,n2;kwargs...)

	s_arr = String[]

	δτ, = δτ_isotropic_δc_int_K_δc(n1,n2,los;hω=h_ω,kwargs...)
	s = @sprintf "%-50s %g" "First Born, ∫dr δc(r) K(r)" round(δτ,sigdigits=3)
	push!(s_arr,s)

	δτ, = δτ_isotropic_δc_int_hω_δCω_FB(n1,n2,los;hω=h_ω,kwargs...)
	s = @sprintf "%-50s %g" "First Born, ∫dω/2π h(ω) δC(ω)" round(δτ,sigdigits=3)
	push!(s_arr,s)

	δτ,δCω = δτ_isotropic_δc_int_hω_δCω_CmC0(n1,n2,los;hω=h_ω,kwargs...)
	s = @sprintf "%-50s %g" "C-C0, ∫dω/2π h(ω) δC(ω)" round(δτ,sigdigits=3)
	push!(s_arr,s)

	δτ, = δτ_isotropic_δc_int_ht_δCt_CmC0(n1,n2,los;ht=h_t,Cω=C_ω,kwargs...,δCω=δCω)
	s = @sprintf "%-50s %g" "C-C0, ∫dt h(t) δC_R(t)" round(δτ,sigdigits=3)
	push!(s_arr,s)

	println("\nTraveltimes\n")
	for s in s_arr
		println(s)
	end
end

function δτ_Δϕ(::soundspeed,los::los_direction=los_radial();kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") Nt dt dν Nν dω ν_start_zeros Nν_Gfn

	# ϕ in degrees
	ϕ_low = get(kwargs,:ϕ_low,30)
	ϕ_high = get(kwargs,:ϕ_high,60)
	nϕ=get(kwargs,:nϕ,2)
	ϕ2_deg = LinRange(ϕ_low,ϕ_high,nϕ)
	ϕ2_arr = ϕ2_deg.*π/180
	n1 = Point2D(Equator(),0)
	n2_arr = [Point2D(Equator(),ϕ2) for ϕ2 in ϕ2_arr]

	t_inds = Vector{UnitRange}(undef,length(n2_arr))

	δτ_δc_CmC0 = zeros(length(n2_arr))

	C_ω = Cω(n1,n2_arr,los;kwargs...,c_scale=1)
	hω_arr = similar(C_ω)
	C′_ω = Cω(n1,n2_arr,los;kwargs...,c_scale=get(kwargs,:c_scale,1+1e-5))

	for (n2ind,n2) in enumerate(n2_arr)
		h = hω(TravelTimes(),view(C_ω,:,n2ind),n1,n2;kwargs...)
		hω_arr[axes(h,1) .+ ν_start_zeros,n2ind] .= parent(h)
	end

	δτ_δc_CmC0 =  vec(2sum((@. real(conj(hω_arr) * (C′_ω - C_ω) ) ),dims=1)*dω/2π)

	hω_arr = hω_arr[ν_start_zeros .+ (1:Nν_Gfn),:]
	δτ_δc_FB = δτ_isotropic_δc_int_K_δc(n1,n2_arr,los;hω=hω_arr,kwargs...)

	header = FITSHeader(["phi_low","phi_high"],[ϕ_low,ϕ_high],
		["Minimum azimuth","Maximum azimuth"])

	los_tag = los_tag_string(los)
	filename = joinpath(SCRATCH_kerneldir,"δτ_δc_CmC0"*los_tag*".fits")
	FITS(filename,"w") do f
		write(f,δτ_δc_CmC0,header=header)
	end

	filename = joinpath(SCRATCH_kerneldir,"δτ_δc_FB"*los_tag*".fits")
	FITS(filename,"w") do f
		write(f,δτ_δc_FB,header=header)
	end

	percentage_diff = @. (δτ_δc_FB-δτ_δc_CmC0)/δτ_δc_CmC0*100
	display(hcat(collect(ϕ2_deg),δτ_δc_FB,δτ_δc_CmC0,percentage_diff))

	δτ_δc_FB,δτ_δc_CmC0
end

end # module