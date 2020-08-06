module Amplitudes

using ..Kernel

# Flows

uniform_rotation_uplus(Ω_rot=20e2/Rsun) = @. √(4π/3)*im*Ω_rot*r

function A_uniform_rotation_int_K_u_FB(xobs1,xobs2;kwargs...)
	K₊ = kernel_ℑu⁺₁₀(Amplitudes(),xobs1,xobs2;kwargs...)
	
	Ω_rot=get(kwargs,:Ω_rot,20e2/Rsun)
	u⁺ = uniform_rotation_uplus(Ω_rot)

	A_v_FB = simps((@.r^2 * K₊*imag(u⁺)),r)
	@save_to_fits_and_return(A_v_FB)
end

function A_uniform_rotation_int_hω_δCω_FB(n1,n2;kwargs...)
	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dω ν_start_zeros Nν_Gfn

	h_ω = get(kwargs,:hω) do 
		hω(Amplitudes(),n1,n2;kwargs...)
	end

	δCω = δCω_uniform_rotation_firstborn_integrated_over_angle(n1,n2;
				kwargs...)[ν_start_zeros .+ axes(h_ω,1)]

	A = sum((@. 2real(conj(h_ω)*δCω)))* dω/2π
end

function A_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1,n2;kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dω ν_start_zeros Nν_Gfn

	h_ω = get(kwargs,:hω) do 
		hω(Amplitudes(),n1,n2;kwargs...)
	end

	δCω = δCω_uniform_rotation_rotatedwaves_linearapprox(n1,n2;
		kwargs...)[ν_start_zeros .+ axes(h_ω,1)]

	A = sum((@. 2real(conj(h_ω)*δCω)))*dω/2π
end

function A_uniform_rotation_rotatedframe_int_hω_δCω(n1,n2;kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dω ν_start_zeros Nν_Gfn

	h_ω = get(kwargs,:hω) do 
		hω(Amplitudes(),n1,n2;kwargs...)
	end

	hω = hω[ν_start_zeros .+ (1:Nν_Gfn)] # only in range

	δCω = δCω_uniform_rotation_rotatedwaves(n1,n2;
		kwargs...)[ν_start_zeros .+ (1:Nν_Gfn)]

	A = sum((@. 2real(conj(h_ω)*δCω)))*dω/2π
end

function A_uniform_rotation_rotatedframe_int_ht_δCt(n1,n2;kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt dν

	Cω_n1n2 = get(kwargs,:Cω) do 
		Cω(n1,n2;kwargs...)
	end
	Ct_n1n2 = get(kwargs,:Ct) do 
		Ct(Cω_n1n2;kwargs...,dν=dν)
	end

	τ_ind_range = get(kwargs,:τ_ind_range) do 
		time_window_indices_by_fitting_bounce_peak(Ct_n1n2,n1,n2;
			dt=dt,Nt=Nt,kwargs...)
	end

	ht = get(kwargs,:ht) do 
		ht(Amplitudes(),Cω_n1n2,n1,n2;τ_ind_range=τ_ind_range,kwargs...)
	end
	ht = ht[τ_ind_range]

	δC_t = δCt_uniform_rotation_rotatedwaves(n1,n2;τ_ind_range=τ_ind_range,kwargs...)

	A = simps((ht.*parent(δC_t)),dt)
end

function A_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1,n2;kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt dν Nt

	if isnothing(get(kwargs,:Cω,nothing)) && isnothing(get(kwargs,:∂ϕ₂Cω,nothing))
		Cω_n1n2,∂ϕ₂Cω_n1n2 = Cω_∂ϕ₂Cω(n1,n2;kwargs...)
	else
		Cω_n1n2 = get(kwargs,:Cω) do 
					Cω(n1,n2;kwargs...)
			end
		∂ϕ₂Cω_n1n2 = get(kwargs,:∂ϕ₂Cω) do 
				∂ϕ₂Cω(n1,n2;kwargs...)
			end
	end

	τ_ind_range = get(kwargs,:τ_ind_range) do 
		time_window_indices_by_fitting_bounce_peak(Ct(Cω_n1n2,dν=dν),
							n1,n2;dt=dt,Nt=Nt,kwargs...)
	end

	ht = get(kwargs,:ht) do
		ht(Amplitudes(),Cω_n1n2,n1,n2;τ_ind_range=τ_ind_range,kwargs...)
	end

	∂ϕ₂Ct_n1n2 = ∂ϕ₂Ct(∂ϕ₂Cω_n1n2,dν=dν)
	δC_t = δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_n1n2;kwargs...)
	
	A = simps((@. ht[τ_ind_range]*δC_t[τ_ind_range]),dt)
end

function A_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(
	n1::Point2D,n2_arr::Vector{<:Point2D};kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt dν Nt

	Cω_n1n2 = get(kwargs,:Cω) do 
				Cω(n1,n2_arr;kwargs...)
			end

	∂ϕ₂Cω_n1n2 = get(kwargs,:∂ϕ₂Cω) do 
					∂ϕ₂Cω(n1,n2_arr;kwargs...)
				end
	
	τ_ind_range = get(kwargs,:τ_ind_range) do
					time_window_indices_by_fitting_bounce_peak(
					Ct(Cω_n1n2,dν=dν),n1,n2_arr;dt=dt,Nt=Nt,kwargs...)
	end

	ht = get(kwargs,:ht) do 
		ht(Amplitudes(),Cω_n1n2,n1,n2;τ_ind_range=τ_ind_range,kwargs...)
	end

	∂ϕ₂Ct_n1n2 = ∂ϕ₂Ct(∂ϕ₂Cω_n1n2,dν=dν)
	δC_t = δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_n1n2;kwargs...)

	A = zeros(length(n2_arr))
	for (n2ind,τ_inds) in enumerate(τ_ind_range)
		A[n2ind] = simps((@. ht[τ_inds,n2ind]*δC_t[τ_inds,n2ind]),dt)
	end
	return A
end

function validate(::flows,n1,n2,los::los_radial=los_radial();kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt dν

	C_ω,∂ϕ₂_Cω = Cω_∂ϕ₂Cω(n1,n2,los;kwargs...)
	h_t = ht(Amplitudes(),C_ω,n1,n2;kwargs...)
	h_ω = hω(Amplitudes(),C_ω,n1,n2;kwargs...)

	A1 = A_uniform_rotation_int_K_u_FB(n1,n2;hω=h_ω,kwargs...)
	s1 = @sprintf "%-50s %g" "First Born, ∫dr u(r) K(r)" round(A1,sigdigits=3)

	A2 = A_uniform_rotation_int_hω_δCω_FB(n1,n2;hω=h_ω,kwargs...)
	s2 = @sprintf "%-50s %g" "First Born, ∫dω/2π h(ω) δC_FB(ω)" round(A2,sigdigits=3)

	A3 = A_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1,n2;hω=h_ω,
		Cω=C_ω,∂ϕ₂Cω=∂ϕ₂_Cω,kwargs...)
	s3 = @sprintf "%-50s %g" "Rotated frame, ∫dω/2π h(ω) δC_R_lin(ω)" round(A3,sigdigits=3)

	A4 = A_uniform_rotation_rotatedframe_int_ht_δCt(n1,n2;ht=h_t,Cω=C_ω,kwargs...)
	s4 = @sprintf "%-50s %g" "Rotated frame, ∫dt h(t) δC_R(t)" round(A4,sigdigits=3)

	A5 = A_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1,n2;ht=h_t,
		Cω=C_ω,∂ϕ₂Cω=∂ϕ₂_Cω,kwargs...)
	s5 = @sprintf "%-50s %g" "Rotated frame, ∫dt h(t) δC_R_lin(t)" round(A5,sigdigits=3)

	println("\nAmplitudes\n")
	for s in [s1,s2,s3,s4,s5]
		println(s)
	end
end

function A_Δϕ(::flows,los::los_radial=los_radial();kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = crosscov.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") Nt dt dν Nν dω

	# ϕ in degrees
	ϕ_low = get(kwargs,:ϕ_low,40)
	ϕ_high = get(kwargs,:ϕ_high,75)
	nϕ=get(kwargs,:nϕ,5)
	ϕ2_deg = LinRange(ϕ_low,ϕ_high,nϕ)
	ϕ2_arr = ϕ2_deg.*π/180
	n1 = Point2D(π/2,0)
	n2_arr = [Point2D(π/2,ϕ2) for ϕ2 in ϕ2_arr]

	t_inds = Vector{UnitRange}(undef,length(n2_arr))

	hω_arr = zeros(ComplexF64,1:Nν,length(n2_arr))

	bounce_no = get(kwargs,:bounce_no,1)

	A_v_CmC0 = zeros(length(n2_arr))
	
	@showprogress 1 "Computing hω_A" for (n2ind,n2) in enumerate(n2_arr)
		C_ω = Cω(n1,n2,los;kwargs...)
		C_t = Ct(C_ω,dν=dν)
		t_inds[n2ind] = crosscov.time_window_indices_by_fitting_bounce_peak(
						C_t,n1,n2,dt=dt,Nt=Nt,bounce_no=bounce_no)

		hω_arr[:,n2ind] .= hω(Amplitudes(),C_ω,n1,n2;τ_ind_range=t_inds[n2ind],kwargs...)
	end

	δC_ω = δCω_uniform_rotation_rotatedwaves(n1,n2_arr,los;hω=h_ω_arr,kwargs...)

	A_v_CmC0 =  2sum((@. real(conj(hω) * δC_ω ) ))*dω/2π

	A_v_CmC0 = @save_to_fits_and_return(A_v_CmC0,los)

	A_v_FB = A_uniform_rotation_int_K_u_FB(n1,n2_arr,los;hω=h_ω_arr,kwargs...)

	A_v_FB = @save_to_fits_and_return(A_v_FB,los)

	A_v_FB,A_v_CmC0
end

# Sound speed

# Y₀₀ = 1/√(4π), so δc₀₀(r) = √(4π) δc(r)
δc₀₀(δc_scale=1e-5) = @. δc_scale*c * √(4π)

function A_isotropic_δc_int_K_δc(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	K = kernel_δc₀₀(Amplitudes(),xobs1,xobs2,los;kwargs...)
	c_scale = get(kwargs,:c_scale,1+1e-5)
	δc = δc₀₀(c_scale-1)

	A_δc_FB = simps((@. r^2 *K* δc),r)
	@save_to_fits_and_return(A_δc_FB,los)
end

function A_isotropic_δc_int_hω_δCω_FB(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dω ν_start_zeros Nν_Gfn

	h_ω = get(kwargs,:hω) do 
		hω(Amplitudes(),xobs1,xobs2,los;kwargs...)
	end

	c_scale = get(kwargs,:c_scale,1+1e-5)

	δCω = δCω_isotropicδc_firstborn_integrated_over_angle(xobs1,xobs2,los;
		kwargs...,c_scale=c_scale)

	δCω = δCω[ν_start_zeros .+ axes(h_ω,1)]

	A = sum(@. 2real(conj(h_ω)*δCω))*dω/2π
end

function A_isotropic_δc_int_ht_δCt_CmC0(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)
	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt dν
	c_scale = get(kwargs,:c_scale,1+1e-5)
	bounce_no = get(kwargs,:bounce_no,1)

	Cω_xobs1xobs2 = Cω(xobs1,xobs2,los;kwargs...,c_scale=1)
	Ct_xobs1xobs2 = Ct(Cω_xobs1xobs2,dν=dν)
	τ_ind_range = time_window_indices_by_fitting_bounce_peak(Ct_xobs1xobs2,xobs1,xobs2,
					dt=dt,Nt=Nt,bounce_no=bounce_no)

	ht = get(kwargs,:ht) do 
		ht(Amplitudes(),Cω_xobs1xobs2,xobs1,xobs2;τ_ind_range=τ_ind_range,kwargs...)
	end

	δCω = get(kwargs,:δCω,nothing)
	δCt = if isnothing(δCω)
		C′t_xobs1xobs2 = Ct(xobs1,xobs2,los;kwargs...,c_scale=c_scale)
		δCt = @. C′t_xobs1xobs2 - Ct_xobs1xobs2
	else
		δCt = δCt_isotropicδc_C_minus_C0(δCω;kwargs...)
	end

	A_δc_CmC0 = simps((@. ht *δCt),dt)
	@save_to_fits_and_return(A_δc_CmC0,los)
end

function A_isotropic_δc_int_hω_δCω_CmC0(xobs1,xobs2,
	los::los_direction=los_radial();kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dω ν_start_zeros Nν_Gfn

	c_scale = get(kwargs,:c_scale,1+1e-5)
	bounce_no = get(kwargs,:bounce_no,1)

	h_ω = get(kwargs,:hω) do 
		hω(Amplitudes(),xobs1,xobs2,los;bounce_no=bounce_no,kwargs...)
	end

	δCω = δCω_isotropicδc_C_minus_C0(xobs1,xobs2,los;kwargs...)
	δCω_in_range = δCω[ν_start_zeros .+ axes(h_ω,1)]

	A = sum(@. 2real(conj(h_ω)*δCω_in_range))*dω/2π
	A,δCω
end

function validate(::soundspeed,n1,n2,los::los_direction=los_radial();kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt dν

	C_ω = Cω(n1,n2,los;kwargs...)
	h_t = ht(Amplitudes(),C_ω,n1,n2;kwargs...)
	h_ω = hω(Amplitudes(),C_ω,n1,n2;kwargs...)

	A1 = A_isotropic_δc_int_K_δc(n1,n2,los;hω=h_ω,kwargs...)
	s1 = @sprintf "%-50s %g" "First Born, ∫dr δc(r) K(r)" round(A1,sigdigits=3)

	A2 = A_isotropic_δc_int_hω_δCω_FB(n1,n2,los;hω=h_ω,kwargs...)
	s2 = @sprintf "%-50s %g" "First Born, ∫dω/2π h(ω) δC(ω)" round(A2,sigdigits=3)

	A3,δCω = A_isotropic_δc_int_hω_δCω_CmC0(n1,n2,los;hω=h_ω,kwargs...)
	s3 = @sprintf "%-50s %g" "C-C0, ∫dω/2π h(ω) δC(ω)" round(A3,sigdigits=3)

	A4 = A_isotropic_δc_int_ht_δCt_CmC0(n1,n2,los;ht=h_t,δCω=δCω,kwargs...)
	s4 = @sprintf "%-50s %g" "C-C0, ∫dt h(t) δC_R(t)" round(A4,sigdigits=3)

	println("\nAmplitudes\n")
	for s in [s1,s2,s3,s4]
		println(s)
	end
end

function A_Δϕ(::soundspeed,los::los_direction=los_radial();kwargs...)

	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") Nt dt dν Nν dω

	# ϕ in degrees
	ϕ_low = get(kwargs,:ϕ_low,30)
	ϕ_high = get(kwargs,:ϕ_high,60)
	nϕ=get(kwargs,:nϕ,2)
	ϕ2_deg = LinRange(ϕ_low,ϕ_high,nϕ)
	ϕ2_arr = ϕ2_deg.*π/180
	n1 = Point2D(π/2,0)
	n2_arr = [Point2D(π/2,ϕ2) for ϕ2 in ϕ2_arr]

	t_inds = Vector{UnitRange}(undef,length(n2_arr))

	hω_arr = zeros(ComplexF64,1:Nν,length(n2_arr))

	bounce_no = get(kwargs,:bounce_no,1)

	A_δc_CmC0 = zeros(length(n2_arr))

	p = Progress(5, 1,"Computing A : ")

	C_ω = Cω(n1,n2_arr,los;kwargs...,c_scale=1)
	next!(p)
	C′_ω = Cω(n1,n2_arr,los;kwargs...,c_scale=get(kwargs,:c_scale,1+1e-5))
	next!(p)

	for (n2ind,n2) in enumerate(n2_arr)
		hω_arr[:,n2ind] .= hω_A(view(C_ω,:,n2ind),n1,n2;kwargs...)
	end
	next!(p)

	A_δc_CmC0 =  vec(2sum((@. real(conj(hω_arr) * (C′_ω - C_ω) ) ),dims=1)*dω/2π)
	next!(p)

	A_δc_FB = A_isotropic_δc_int_K_δc(n1,n2_arr,los;hω=h_ω_arr,kwargs...)
	next!(p)

	A_δc_CmC0 = @save_to_fits_and_return(A_δc_CmC0,los)

	A_δc_FB = @save_to_fits_and_return(A_δc_FB,los)

	A_δc_FB,A_δc_CmC0
end

end # module