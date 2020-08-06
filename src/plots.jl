module Plots

using CSV
using DataFrames
using DelimitedFiles
using Distributed
using LaTeXStrings
using LinearAlgebra
using FFTW
using FITSIO
using JLD2
using LsqFit
using NPZ
using OffsetArrays
using PointsOnASphere
using Polynomials
using Printf
using ProgressMeter
using PyCall
using PyPlot
using SphericalHarmonicModes

@everywhere begin 
	using ParallelUtilities
	using NumericallyIntegrateArrays
	using DistributedArrays
end

pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
ticker =  pyimport("matplotlib.ticker")
colors = pyimport("matplotlib.colors")
cm = pyimport("matplotlib.cm")
# ho_endpt_so = pyimport("ho_endpt_so")
axes_grid1 = pyimport("mpl_toolkits.axes_grid1")

import ..Crosscov: SCRATCH_kerneldir, SCRATCH, @append_los_if_necessary
import ..Greenfn_radial: Rsun, r, c, ρ, nr, r_src_default, r_obs_default
const r_frac = r./Rsun;

import ..Directions: los_radial, los_earth

los_tag_string(::los_radial) = ""
los_tag_string(::los_earth) = "_los"

# traveltimes

function plot_δτ_validation_v(los = los_radial();kwargs...)
	
	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = crosscov.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") Nt dt dν Nν dω

	los_tag = los_tag_string(los)

	filename = joinpath(SCRATCH_kerneldir,"δτ_v_CmC0$los_tag.fits")
	if isfile(filename)
		δτ_v_CmC0 = FITS(filename,"r") do f
							read(f[1])
						end
	else
		throw(DimensionMismatch("Run δτ_v_compare before plotting"))
	end

	header = FITS(filename,"r") do f
				read_header(f[1])
			end

	ϕ_low = header["PHI_LOW"]
	ϕ_high = header["PHI_HIGH"]

	filename = joinpath(SCRATCH_kerneldir,"δτ_v_FB$los_tag.fits")
	if isfile(filename)
		δτ_v_FB = FITS(filename,"r") do f
					read(f[1])
				end	
	else
		throw(DimensionMismatch("Run δτ_v_compare before plotting"))
	end

	nϕ=get(kwargs,:nϕ,length(δτ_v_CmC0))

	ϕ2_deg = collect(LinRange(ϕ_low,ϕ_high,nϕ))
	ϕ2_arr = deg2rad.(ϕ2_deg)
	n1 = Point2D(π/2,0)
	n2_arr = [Point2D(π/2,ϕ2) for ϕ2 in ϕ2_arr]

	percentage_diff = (δτ_v_CmC0-δτ_v_FB)./δτ_v_CmC0.*100

	ϕ2_arr = rad2deg.(ϕ2_arr)

	ax_δτ = get(kwargs,:ax_δτ) do 
		subplot2grid((3,1),(0,0),rowspan=2)
	end
	ax_δτ.plot(ϕ2_arr,δτ_v_CmC0,label="C - C₀",color="black",ls="dashed")
	ax_δτ.plot(ϕ2_arr,δτ_v_FB,"^",label="First Born",ms=6,ls="None",color="black")
	ax_δτ.set_ylabel("Travel time shift [sec]",fontsize=12)
	ax_δτ.legend(loc="best")
	ax_δτ.xaxis.set_major_formatter(ticker.NullFormatter())
	ax_δτ.xaxis.set_major_locator(ticker.MaxNLocator(5))

	ax_δτ_diff = get(kwargs,:ax_δτ_diff) do
		subplot2grid((3,1),(2,0))
	end
	ax_δτ_diff.plot(ϕ2_arr,percentage_diff,"o-",ms=4,zorder=2,color="black")
	ax_δτ_diff.set_ylabel("Difference",fontsize=12)
	ax_δτ_diff.set_xlabel("Angular separation [degrees]",fontsize=12)
	ax_δτ_diff.axhline(0,ls="dotted",color="black",zorder=0)
	ax_δτ_diff.xaxis.set_major_locator(ticker.MaxNLocator(5))
	ax_δτ_diff.yaxis.set_major_locator(ticker.MaxNLocator(3))
	ax_δτ_diff.margins(y=0.2)
	ax_δτ_diff.set_yticklabels([(@sprintf "%.2f" x)*"%" 
				for x in ax_δτ_diff.get_yticks()])

	gcf().set_size_inches(8,4)
	tight_layout()

	dτ_arr = DataFrame(dist=ϕ2_deg,dt_FB=δτ_v_FB,dt_C=δτ_v_CmC0,
		percentage_diff=percentage_diff)

	save = get(kwargs,:save,true)
	save && savefig(joinpath(SCRATCH,"dt_v$los_tag.eps"))
	save && CSV.write(joinpath(SCRATCH,"dt_v$los_tag"),dτ_arr,delim=' ')

	dτ_arr
end

function plot_δτ_validation_δc(los = los_radial();kwargs...)
	
	r_src = get(kwargs,:r_src,r_src_default)
	Gfn_path_src = crosscov.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") Nt dt dν Nν dω

	los_tag = los_tag_string(los)

	filename = joinpath(SCRATCH_kerneldir,"δτ_δc_CmC0$los_tag.fits")
	if isfile(filename)
		δτ_δc_CmC0 = FITS(filename,"r") do f
							read(f[1])
						end
	else
		throw(DimensionMismatch("Run δτ_δc_compare before plotting"))
	end

	header = FITS(filename,"r") do f
				read_header(f[1])
			end

	ϕ_low = header["PHI_LOW"]
	ϕ_high = header["PHI_HIGH"]

	filename = joinpath(SCRATCH_kerneldir,"δτ_δc_FB$los_tag.fits")
	if isfile(filename)
		δτ_δc_FB = FITS(filename,"r") do f
					read(f[1])
				end	
	else
		throw(DimensionMismatch("Run δτ_δc_compare before plotting"))
	end

	nϕ=get(kwargs,:nϕ,length(δτ_δc_CmC0))

	ϕ2_deg = collect(LinRange(ϕ_low,ϕ_high,nϕ))
	ϕ2_arr = deg2rad.(ϕ2_deg)
	n1 = Point2D(π/2,0)
	n2_arr = [Point2D(π/2,ϕ2) for ϕ2 in ϕ2_arr]

	percentage_diff = (δτ_δc_CmC0-δτ_δc_FB)./δτ_δc_CmC0.*100

	ϕ2_arr = rad2deg.(ϕ2_arr)

	ax_δτ = get(kwargs,:ax_δτ) do 
		subplot2grid((3,1),(0,0),rowspan=2)
	end
	ax_δτ.plot(ϕ2_arr,δτ_δc_CmC0,label="C - C₀",color="black",ls="dashed")
	ax_δτ.plot(ϕ2_arr,δτ_δc_FB,"^",label="First Born",ms=6,ls="None",color="black")
	ax_δτ.set_ylabel("Travel time shift [sec]",fontsize=12)
	ax_δτ.legend(loc="best")
	ax_δτ.xaxis.set_major_formatter(ticker.NullFormatter())
	ax_δτ.xaxis.set_major_locator(ticker.MaxNLocator(5))

	ax_δτ_diff = get(kwargs,:ax_δτ_diff) do
		subplot2grid((3,1),(2,0))
	end
	ax_δτ_diff.plot(ϕ2_arr,percentage_diff,"o-",ms=4,zorder=2,color="black")
	ax_δτ_diff.set_ylabel("Difference",fontsize=12)
	ax_δτ_diff.set_xlabel("Angular separation [degrees]",fontsize=12)
	ax_δτ_diff.axhline(0,ls="dotted",color="black",zorder=0)
	ax_δτ_diff.xaxis.set_major_locator(ticker.MaxNLocator(5))
	ax_δτ_diff.yaxis.set_major_locator(ticker.MaxNLocator(3))
	ax_δτ_diff.margins(y=0.2)
	ax_δτ_diff.set_yticklabels([(@sprintf "%.2f" x)*"%" 
				for x in ax_δτ_diff.get_yticks()])

	tight_layout()

	dτ_arr = DataFrame(dist=ϕ2_deg,dt_FB=δτ_δc_FB,dt_C=δτ_δc_CmC0,
		percentage_diff=percentage_diff)

	save = get(kwargs,:save,false)
	save && savefig(joinpath(SCRATCH,"dt_dc$los_tag.eps"))
	save && CSV.write(joinpath(SCRATCH,"dt_dc$los_tag"),dτ_arr,delim=' ')

	dτ_arr
end

function plot_δτ_validation_δc_with_and_without_los(;kwargs...)
	ax_δτ_radial = subplot2grid((3,2),(0,0),rowspan=2)
	ax_δτ_radial_diff = subplot2grid((3,2),(2,0))
	ax_δτ_los = subplot2grid((3,2),(0,1),rowspan=2)
	ax_δτ_los_diff = subplot2grid((3,2),(2,1))

	plot_δτ_validation_δc(los_radial();kwargs...,
		ax_δτ=ax_δτ_radial,
		ax_δτ_diff=ax_δτ_radial_diff,
		save=false)

	plot_δτ_validation_δc(los_earth();kwargs...,
		ax_δτ=ax_δτ_los,
		ax_δτ_diff=ax_δτ_los_diff,
		save=false)

	ax_δτ_radial.set_title("Radial")
	ax_δτ_los.set_title("Line of sight")

	gcf().set_size_inches(8,4)
	tight_layout()
	savefig(joinpath(SCRATCH,"dt_dc_radial_and_los.eps"))
end

# crosscov

function plot_time_distance_with_and_without_los(;kwargs...) # time in hours

	r_src=get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt

	t_max = get(kwargs,:t_max,5)
	t_min = get(kwargs,:t_min,1)
	ℓ_range = get(kwargs,:ℓ_range,20:100)
	nϕ = get(kwargs,:nϕ,maximum(ℓ_range))

	t_max_ind = floor(Int,t_max*60^2/dt) + 1
	t_min_ind = floor(Int,t_min*60^2/dt) + 1
	t_inds = t_min_ind:t_max_ind

	t = t_inds.*dt./60^2
	ϕ = LinRange(5,90,nϕ).*π/180

	nobs1 = Point2D(π/2,0)

	filename = joinpath(SCRATCH_kerneldir,"CtΔϕ.fits") 
	if isfile(filename)
		C =	FITS(filename,"r") do f
				read(f[1])
			end
		if size(C,1) == Nt
			C = C[t_inds,:]
		end
	else
		C = crosscov.CtΔϕ(τ_ind_range=t_inds,
			ℓ_range=ℓ_range,Δϕ_arr=ϕ)
	end

	filename = joinpath(SCRATCH_kerneldir,"Ct.fits")
	if isfile(filename)
		C60 =	FITS(filename,"r") do f
					read(f[1])
				end
		if size(C60,1) == Nt
			C60 = C60[t_inds]
		end
	else
		C60 = crosscov.Ct(π/3,
			ℓ_range=ℓ_range,τ_ind_range=t_inds)
	end

	filename = joinpath(SCRATCH_kerneldir,"Ct_los.fits")
	if isfile(filename)
		C60_los =	FITS(filename,"r") do f
						read(f[1])
					end
		if size(C60_los,1) == Nt
			C60_los = C60_los[t_inds]
		end
	else
		C60_los = crosscov.Ct_los(nobs1,π/3,
			ℓ_range=ℓ_range,τ_ind_range=t_inds)
	end
	
	ax1 = get(kwargs,:ax1) do 
		subplot2grid((1,3),(0,0),colspan=2)
	end
	ax2 = get(kwargs,:ax2) do
		subplot2grid((1,3),(0,2),colspan=1)
	end

	C_norm = maximum(abs,C)
	ax1.pcolormesh(ϕ.*180/π,t,C./C_norm,
		cmap="Greys",vmax=0.3,vmin=-0.3,rasterized=true)
	ax1.set_xlabel("Angular separation [degrees]",fontsize=12);
	ax1.set_ylabel("Time [hours]",fontsize=12);
	ax1.axvline(60,color="black",ls="dashed",lw=0.6)
	ax1.set_title("Time-distance diagram",fontsize=12);

	ax2.plot(C60,t,lw=0.7,color="grey",label="Radial")
	ax2.plot(C60_los,t,lw=0.7,color="black",label="LoS")
	ax2.yaxis.set_major_formatter(ticker.NullFormatter())
	ax2.set_ylim(ax1.get_ylim())
	ax2.set_ylabel("Time [hours]",fontsize=12);
	ax2.legend(loc="lower left")

	tight_layout()

	save = get(kwargs,:save,true)
	save && savefig(joinpath(SCRATCH,"C_timedistance.eps"))
end

function plot_time_distance_bounce_filtered(;kwargs...)
	r_src = get(kwargs,:r_src,crosscov.r_src_default)
	t_max = get(kwargs,:t_max,6)
	t_min = get(kwargs,:t_min,0.3)
	ℓ_range = get(kwargs,:ℓ_range,20:100)
	nϕ = get(kwargs,:nϕ,maximum(ℓ_range))
	bounce_no = get(kwargs,:bounce_no,1)
	los = get(kwargs,:los,los_radial())
	
	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt

	t_max_ind = floor(Int,t_max*60^2/dt)+1
	t_min_ind = floor(Int,t_min*60^2/dt)+1
	t_inds = t_min_ind:t_max_ind

	t = t_inds.*dt./60^2

	ϕ = LinRange(5,90,nϕ).*π/180

	filename = joinpath(SCRATCH_kerneldir,"CΔϕt.fits")
	if isfile(filename)
		println("Reading from $filename")
		C = FITS(filename,"r") do f
				read(f[1])
			end
		C = C[:,t_inds]
	else
		error("File not found")
	end

	C_filt = copy(C)
	f_t = zeros(size(C_filt))
	for (ind,Δϕ_i) in enumerate(ϕ)
		τ_low_ind,τ_high_ind = crosscov.time_window_bounce_filter(Δϕ_i,dt,bounce_no)
		f_t[τ_low_ind:τ_high_ind,ind] .= 1
	end

	C_filt .*= f_t

	ax1 = get(kwargs,:ax1) do 
		subplot(121)
	end
	ax2 = get(kwargs,:ax2) do 
		subplot(122)
	end

	ax1.pcolormesh(ϕ.*180/π,t,C./maximum(abs,C),
		cmap="Greys",vmax=1,vmin=-1,rasterized=true)
	ax1.set_xlabel("Angular separation [degrees]",fontsize=12);
	ax1.set_ylabel("Time [hours]",fontsize=12);
	ax1.set_title("Time-distance diagram",fontsize=12);

	ax2.pcolormesh(ϕ.*180/π,t,C_filt./maximum(abs,C_filt),
		cmap="Greys",vmax=1,vmin=-1,rasterized=true)
	ax2.set_xlabel("Angular separation [degrees]",fontsize=12);
	ax2.set_ylabel("Time [hours]",fontsize=12);
	ax2.set_title("Filtered bounce",fontsize=12);
end

function plot_C_spectrum(;kwargs...)

	r_src=get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr

	Nν_Gfn = length(ν_arr)

	filename = joinpath(SCRATCH_kerneldir,"Cωℓ_in_range.fits")
	
	Cωℓ,header = FITS(filename,"r") do f
				read(f[1]),read_header(f[1])
			end
	
	ν_ind_range = get(kwargs,:ν_ind_range,axes(Cωℓ,1))
	ℓ_range = get(kwargs,:ℓ_range,axes(Cωℓ,2))

	ℓ_edges = collect(first(ℓ_range):last(ℓ_range)+1)
	ν_edges = collect(ν_arr[first(ν_ind_range):last(ν_ind_range)]).*1e3
	spec = parent(Cωℓ)./maximum(abs,Cωℓ)

	observed_freq = readdlm("Schou_freq181qr.1288")
	l = observed_freq[:,1]; ν=observed_freq[:,3]

	ax = get(kwargs,:ax) do
		subplot(111) 
	end # default is a single subplot

	ax.pcolormesh(ℓ_edges,ν_edges,spec,
		cmap="Greys",vmax=0.1,vmin=0,rasterized=true);
	ax.set_xlabel(L"$\ell$",fontsize=12);
	ax.set_ylabel("Frequency [mHz]",fontsize=12);
	modes_filter = @. (2e3 < ν < 4.5e3) & (1 < l < 100) & ( mod(l,5)==0) 
	ax.plot(l[modes_filter],ν[modes_filter]./1e3,
		label="MDI frequencies",marker="o",ls="None",mfc="0.8",mec="0.3",ms=4)
	ax.legend(loc="lower left")
	ax.set_title("Spectrum of C",fontsize=12)
	tight_layout()

	save = get(kwargs,:save,true)
	save && savefig(joinpath(SCRATCH,"C_spectrum.eps"))
end

function plot_C_spectrum_and_time_distance(;kwargs...)
	
	ax1 = subplot2grid((1,5),(0,0),colspan=2)
	ax2 = subplot2grid((1,5),(0,2),colspan=2)
	ax3 = subplot2grid((1,5),(0,4),colspan=1)

	plot_C_spectrum(;kwargs...,ax=ax1,save=false)
	plot_time_distance_with_and_without_los(;kwargs...,ax1=ax2,ax2=ax3,save=false)

	plt.gcf().set_size_inches(12,4)
	tight_layout()

	save = get(kwargs,:save,true)
	save && savefig(joinpath(SCRATCH,"C_spectrum_timedistance.eps"))
end

function plot_Ct_groups_of_modes(;t_max=6,t_min=0.3)
	r_src=crosscov.Rsun-75e5
	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt

	t_max_ind = floor(Int,t_max*60^2/dt)+1
	t_min_ind = floor(Int,t_min*60^2/dt)+1

	t_inds = t_min_ind:t_max_ind
	t = t_inds.*dt/60^2
	nt = length(t)

	n1 = Point2D(π/2,0)
	n2 = Point2D(π/2,π/3)

	ℓ_ranges = Iterators.partition(20:99,20)

	C = zeros(nt,length(ℓ_ranges)+1)

	for (ind,ℓ_range) in enumerate(ℓ_ranges)
		C[:,ind] = crosscov.Ct(n1,n2,ℓ_range=ℓ_range)[t_inds]
	end

	C[:,end] = crosscov.Ct(n1,n2,ℓ_range=20:99)[t_inds]

	fig,ax = subplots(nrows=size(C,2),ncols=1)
	for axis in ax
		axis.set_xlim(first(t),last(t))
	end

	ax[1].set_title("Cross covariance (Δϕ=60 degrees)")

	for (ind,ℓ_range) in enumerate(ℓ_ranges)
		ax[ind].plot(t,C[:,ind],label="ℓ=$(ℓ_range[1]:ℓ_range[end])")
		ax[ind].xaxis.set_major_formatter(ticker.NullFormatter())
		ax[ind].yaxis.set_major_locator(ticker.MaxNLocator(3))
		ax[ind].legend(loc="upper right",bbox_to_anchor=(1.01,1.5))
	end

	ax[end].plot(t,C[:,end])
	ax[end].set_title("Sum over ℓ")
	ax[end].yaxis.set_major_locator(ticker.MaxNLocator(3))
	
	xlabel("Time [hours]",fontsize=12)
	tight_layout()
	savefig("Ct_lranges.eps")
end

function plot_h(x1,x2;kwargs...)
	
	r_src = get(kwargs,:r_src,kernel.r_src_default)
	Gfn_path_src = crosscov.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_start_zeros ν_arr Nt dt dν Nν_Gfn

	los = get(kwargs,:los,los_radial())
	los_tag = los_tag_string(los)

	filename = joinpath(SCRATCH_kerneldir,"Cω$los_tag.fits")
	if isfile(filename)
		Cω_x1x2 = FITS(filename,"r") do f
					read(f[1])
				end	
	else
		Cω_x1x2 = crosscov.Cω(x1,x2,los;kwargs...)
	end

	C_t = crosscov.fft_ω_to_t(Cω_x1x2,dν)
	τ_ind_range = crosscov.time_window_indices_by_fitting_bounce_peak(
						C_t,x1,x2;dt=dt,kwargs...)
	ht = crosscov.ht(Cω_x1x2,x1,x2;τ_ind_range=τ_ind_range,kwargs...)
	hω = crosscov.fft_t_to_ω(ht,dt)

	subplot(411)
	plot(ν_arr,real(Cω_x1x2[ν_start_zeros .+ (1:Nν_Gfn)]))
	title("C(x₁,x₂,ω)")

	ax2=subplot(412)
	plot((1:Nt).*dt,C_t,color="black")
	axvline(τ_ind_range[1]*dt,ls="solid")
	axvline(τ_ind_range[end]*dt,ls="solid")
	title("C(x₁,x₂,t)")

	subplot(413,sharex=ax2)
	plot((1:Nt).*dt,ht,color="black")
	xlim(0,60^2*6)

	title("h(x₁,x₂,t)")

	subplot(414)
	plot(ν_arr,imag(hω[ν_start_zeros .+ (1:Nν_Gfn)]),label="imag")
	plot(ν_arr,real(hω[ν_start_zeros .+ (1:Nν_Gfn)]),label="real")
	legend(loc="best")
	title("h(x₁,x₂,ω)")

	tight_layout()

	gcf().savefig(joinpath(SCRATCH,"h$los_tag.eps"))
end

function plot_δCω_isotropicδc_with_and_without_los(args...;kwargs...)

	r_src=get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ν_start_zeros Nν_Gfn

	ν_zoom_min = get(kwargs,:ν_zoom_min,2.985e-3)
	ν_zoom_max = get(kwargs,:ν_zoom_max,3.05e-3)
	ν_zoom_inds = @. ν_zoom_min < ν_arr < ν_zoom_max
	ν_arr_zoom = ν_arr[ν_zoom_inds]

	ax_radial = subplot(121)
	ax_los = subplot(122)
	inset_coords = [0.7, 0.1, 0.25, 0.25]
	ax_radial_zoom = ax_radial.inset_axes(inset_coords)
	ax_los_zoom = ax_los.inset_axes(inset_coords)

	filename = joinpath(SCRATCH_kerneldir,"δCω_isotropicδc_FB.fits")
	if isfile(filename)
		δCω_FB = FITS(filename,"r") do f
					read(f[1])
				end 
		δCω_FB = reinterpret(ComplexF64,δCω_FB)
	else
		δCω_FB = crosscov.δCω_isotropicδc_firstborn_integrated_over_angle(
			args...;kwargs...)
	end
	δCω_FB = δCω_FB[ν_start_zeros .+ (1:Nν_Gfn)]

	filename = joinpath(SCRATCH_kerneldir,"δCω_isotropicδc_CmC0.fits")
	if isfile(filename)
		δCω_CmC₀ = FITS(filename,"r") do f
					read(f[1])
				end 
		δCω_CmC₀ = reinterpret(ComplexF64,δCω_CmC₀)
	else
		δCω_CmC₀ = crosscov.δCω_isotropicδc_C_minus_C0(
			args...;kwargs...)
	end
	δCω_CmC₀ = δCω_CmC₀[ν_start_zeros .+ (1:Nν_Gfn)]

	ax_radial.plot(ν_arr.*1e3,real(δCω_FB),label="First Born",color="grey")
	ax_radial.plot(ν_arr.*1e3,real(δCω_CmC₀),label="C - C₀",
			ls="None",marker=".",color="black")
	ax_radial.set_xlabel("Frequency [mHz]",fontsize=12)
	ax_radial.set_ylabel("ℜ[C]",fontsize=12)
	ax_radial.legend(loc="best")
	ax_radial.set_title("Radial",fontsize=12)

	ax_radial_zoom.plot(ν_arr_zoom.*1e3,real(δCω_FB[ν_zoom_inds]),
		label="First Born",color="grey")
	ax_radial_zoom.plot(ν_arr_zoom.*1e3,real(δCω_CmC₀[ν_zoom_inds]),
		label="C - C₀",ls="None",marker=".",color="black")
	ax_radial_zoom.xaxis.set_major_locator(ticker.MaxNLocator(2))
	ax_radial.indicate_inset_zoom(ax_radial_zoom)

	filename = joinpath(SCRATCH_kerneldir,"δCω_isotropicδc_FB_los.fits")
	if isfile(filename)
		δCω_FB_los = FITS(filename,"r") do f
						read(f[1])
					end 
		δCω_FB_los = reinterpret(ComplexF64,δCω_FB_los)
	else
		δCω_FB_los = crosscov.δCω_isotropicδc_firstborn_integrated_over_angle_los(
			args...;kwargs...)
	end
	δCω_FB_los = δCω_FB_los[ν_start_zeros .+ (1:Nν_Gfn)]

	filename = joinpath(SCRATCH_kerneldir,"δCω_isotropicδc_CmC0_los.fits")
	if isfile(filename)
		δCω_CmC₀_los = FITS(filename,"r") do f
						read(f[1])
					end
		δCω_CmC₀_los = reinterpret(ComplexF64,δCω_CmC₀_los)
	else
		δCω_CmC₀_los = crosscov.δCω_isotropicδc_C_minus_C0(
			args...;kwargs...)
	end
	δCω_CmC₀_los = δCω_CmC₀_los[ν_start_zeros .+ (1:Nν_Gfn)]

	ax_los.plot(ν_arr.*1e3,real(δCω_FB_los),label="First Born",color="grey")
	ax_los.plot(ν_arr.*1e3,real(δCω_CmC₀_los),label="C - C₀",
				ls="None",marker=".",color="black")
	ax_los.set_xlabel("Frequency [mHz]",fontsize=12)
	ax_los.set_ylabel("ℜ[C]",fontsize=12)
	ax_los.legend(loc="best")
	ax_los.set_title("Line-of-sight",fontsize=12)

	ax_los_zoom.plot(ν_arr_zoom.*1e3,real(δCω_FB_los[ν_zoom_inds]),
		label="First Born",color="grey")
	ax_los_zoom.plot(ν_arr_zoom.*1e3,real(δCω_CmC₀_los[ν_zoom_inds]),
		label="C - C₀",ls="None",marker=".",color="black")
	ax_los_zoom.xaxis.set_major_locator(ticker.MaxNLocator(2))
	ax_los.indicate_inset_zoom(ax_los_zoom)

	gcf().set_size_inches(8,4)
	tight_layout()
	save = get(kwargs,:save,true)
	save && savefig(joinpath(SCRATCH,"dCw_dc_with_and_without_los.eps"))
end

function plot_Ct_with_and_without_los(xobs1,xobs2;kwargs...)
	r_src = r_src_default
	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load(joinpath(Gfn_path_src,"parameters.jld2"),Nt,dt)

	τ_ind_range = 1:div(Nt,2)
	t = τ_ind_range.*dt ./60^2

	Ct_los = crosscov.Ct_los(xobs1,xobs2;τ_ind_range=τ_ind_range,kwargs...)
	Ct = crosscov.Ct(xobs1,xobs2;τ_ind_range=τ_ind_range,kwargs...)

	plot(t,parent(Ct),label="radial",color="grey")
	plot(t,parent(Ct_los),label="line-of-sight",color="black")

	xlim(2,5)
	xlabel("Time [hours]",fontsize=12)
	ylabel("Cross covariance",fontsize=12)
	legend(loc="best",fontsize=12)
end

function plot_δCω_uniform_rotation_FB_rotatedframe(;kwargs...)

	r_src=get(kwargs,:r_src,r_src_default)
	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ν_start_zeros Nν_Gfn

	ν_zoom_min = 2.985e-3
	ν_zoom_max = 3.05e-3
	ν_zoom_inds = @. ν_zoom_min < ν_arr < ν_zoom_max
	ν_arr_zoom = ν_arr[ν_zoom_inds]


	filename = joinpath(SCRATCH_kerneldir,"δCω_flows_FB.fits")
	δCω_FB = FITS(filename,"r") do f
		copy(reinterpret(ComplexF64,read(f[1])))
	end

	δCω_FB = δCω_FB[ν_start_zeros .+ (1:Nν_Gfn)]

	filename = joinpath(SCRATCH_kerneldir,"δCω_flows_rotated.fits")
	δCω_rot = FITS(filename,"r") do f
		copy(reinterpret(ComplexF64,read(f[1])))
	end

	δCω_rot = δCω_rot[ν_start_zeros .+ (1:Nν_Gfn)]

	ax_radial = subplot(111)

	ax_radial.plot(ν_arr.*1e3,imag(δCω_FB),
		color="gray",label="First Born")
	ax_radial.plot(ν_arr.*1e3,imag(δCω_rot),
		color="black",ls="None",marker=".",label="Solid-body rotation")
	ax_radial.set_xlabel("Frequency [mHz]",fontsize=12)
	ax_radial.set_ylabel("ℑ[C]",fontsize=12)
	ax_radial.legend(loc="best")
	ax_radial.set_title("Radial",fontsize=12)

	inset_coords = [0.7, 0.1, 0.25, 0.25]
	ax_radial_zoom = ax_radial.inset_axes(inset_coords)

	ax_radial_zoom.plot(ν_arr_zoom.*1e3,imag(δCω_FB[ν_zoom_inds]),
		color="grey")
	ax_radial_zoom.plot(ν_arr_zoom.*1e3,imag(δCω_rot[ν_zoom_inds]),
		ls="None",marker=".",color="black")
	ax_radial_zoom.xaxis.set_major_locator(ticker.MaxNLocator(2))
	ax_radial.indicate_inset_zoom(ax_radial_zoom)

	tight_layout()
	save = get(kwargs,:save,true)
	save && savefig(joinpath(SCRATCH,"dCw_v_FB_rotated.eps"))
end

# Kernels

function compute_Kv_10_different_arrival_bounces()
	n1 = Point2D(π/2,0); n2 = Point2D(π/2,π/3)
	kernels = zeros(kernel.nr,3)

	for (ind,bounce_no) in enumerate((1,2,4))
		kernels[:,ind] .= imag.(kernel.kernel_uniform_rotation_uplus(n1,n2,ℓ_range=20:100,bounce_no=bounce_no))
	end

	mkpath("$SCRATCH/kernels")
	f = FITS("$SCRATCH/kernels/K10_different_bounces.fits","w")
	write(f,kernels)
	close(f)

	return kernels
end

function plot_Kv_10_different_arrival_bounces()

	kernel_file = "$SCRATCH/kernels/K10_different_bounces.fits"
	if isfile(kernel_file)
		f = FITS(kernel_file)
		kernels = read(f[1])
		close(f)
	else
		kernels = compute_K10_different_arrival_bounces()
	end
	
	subplot(211)
	plot(r_frac,normalize(kernels[:,1],Inf),label="first bounce")
	xlim(0.8,r[end])
	legend(loc="best")
	title(L"Normalized $K_{10}(r)$",fontsize=12)

	subplot(212)
	plot(r_frac,kernels[:,3],label="fourth bounce")
	xlim(0.8,r[end])
	legend(loc="best")
	
	xlabel(L"$r/R_\odot$",fontsize=12)
	

	tight_layout()
end

function plot_Kvrₗ₀(;kwargs...)
	kernel_file = joinpath(SCRATCH_kerneldir,"Kₗ₀_δτ_u_rθϕ.fits")
	
	kernel,header = FITS(kernel_file,"r") do f
				read(f[1],:,1,:),read_header(f[1])
	end
	kernel_modes = LM(header["L_MIN"]:header["L_MAX"],0:0)
	s_min = kernel_modes.l_min
	s_max = kernel_modes.l_max

	ls_arr = ["solid","dashed"]
	markers = ["None","."]
	lw = [2,1]
	colors = ["red","blue","green","orange"]

	ax1 = get(kwargs,:ax) do 
		subplot(111)
	end

	inset_coords = [0.1, 0.1, 0.5, 0.3]
	ax1_zoom = ax1.inset_axes(inset_coords)

	for s=s_min:s_max
		s_ind = modeindex(kernel_modes,s,0)
		for ax in [ax1,ax1_zoom]
			ax.plot(r_frac,kernel[:,s_ind],label="ℓ=$s",
					ls=ls_arr[div(s-1,4)+1],
					color=colors[mod(div(s-1,2),4)+1],
					marker=markers[div(s-1,4)+1],
					markevery=40,linewidth=lw[div(s-1,4)+1])
		end
	end

	ax1.set_ylabel(L"\mathrm{K}_{v_r,ℓ0}",fontsize=14)
	ax1.legend(loc="best")
	ax1.set_xlim(0.2,r_frac[end])
	ax1.yaxis.set_major_locator(ticker.MaxNLocator(4))

	ax1_zoom.set_xlim(0.97,r_frac[end])
	ax1_zoom.xaxis.set_major_locator(ticker.MaxNLocator(4))

	ax1.indicate_inset_zoom(ax1_zoom)
	gcf().set_size_inches(8,4)
	tight_layout()
	save = get(kwargs,:save,true)
	save && savefig(joinpath(SCRATCH,"Kvr.eps"),dpi=320)
end

function plot_Kvθₗ₀(;kwargs...)
	
	kernel_file = joinpath(SCRATCH_kerneldir,"Kₗ₀_δτ_u_rθϕ.fits")
	
	kernel,header = FITS(kernel_file,"r") do f
				read(f[1],:,2,:),read_header(f[1])
	end
	kernel_modes = LM(header["L_MIN"]:header["L_MAX"],0:0)
	s_min = kernel_modes.l_min
	s_max = kernel_modes.l_max

	ls_arr = ["solid","dashed"]
	markers = ["None","."]
	lw = [2,1]
	colors = ["red","blue","green","orange"]

	ax2 = get(kwargs,:ax) do 
		subplot(111)
	end

	inset_coords = [0.1, 0.1, 0.5, 0.3]
	ax2_zoom = ax2.inset_axes(inset_coords)

	for s=s_min:s_max
		s_ind = modeindex(kernel_modes,s,0)
		for ax in [ax2,ax2_zoom]
			ax.plot(r_frac,kernel[:,s_ind],label="ℓ=$s")
		end
	end
	ax2.legend(loc="upper left")
	ax2.set_xlim(0.2,r_frac[end])
	ax2.set_ylabel(L"$\mathrm{K}_{v_θ,ℓ0}$",fontsize=14)
	ax2.set_xlabel(L"$r/R_\odot$",fontsize=12)
	ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))

	ax2_zoom.set_xlim(0.97,r_frac[end])
	ax2_zoom.xaxis.set_major_locator(ticker.MaxNLocator(4))

	ax2.indicate_inset_zoom(ax2_zoom)

	gcf().set_size_inches(8,4)
	tight_layout()
	save = get(kwargs,:save,true)
	save && savefig(joinpath(SCRATCH,"Kvtheta.eps"),dpi=320)
end

function plot_Kvϕₗ₀(;kwargs...)
	kernel_file = joinpath(SCRATCH_kerneldir,"Kₗ₀_δτ_u_rθϕ.fits")
	
	kernel,header = FITS(kernel_file,"r") do f
				read(f[1],:,3,:),read_header(f[1])
			end
	kernel_modes = LM(header["L_MIN"]:header["L_MAX"],0:0)
	s_min = kernel_modes.l_min
	s_max = kernel_modes.l_max

	ax1 = get(kwargs,:ax) do 
		subplot(111)
	end

	inset_coords = [0.1, 0.1, 0.5, 0.3]
	ax1_zoom = ax1.inset_axes(inset_coords)

	ls_arr = ["solid","dashed"]
	markers = ["None","."]
	lw = [2,1]
	colors = ["red","blue","green","orange"]

	for s=s_min:s_max
		s_ind = modeindex(kernel_modes,s,0)
		for ax in [ax1,ax1_zoom]
			ax.plot(r_frac,kernel[:,s_ind],label="ℓ=$s")
		end
	end
	ax1.set_ylabel(L"$\mathrm{K}_{vϕ,ℓ0}$",fontsize=14)
	ax1.legend(loc="upper left")
	ax1.set_xlim(0.2,r_frac[end])
	ax1.set_xlabel(L"$r/R_\odot$",fontsize=12)
	ax1.axhline(0,color="black",ls="dotted",zorder=1)
	ax1.yaxis.set_major_locator(ticker.MaxNLocator(4))

	ax1_zoom.set_xlim(0.95,r_frac[end])

	ax1.indicate_inset_zoom(ax1_zoom)

	gcf().set_size_inches(8,4)
	tight_layout()
	save = get(kwargs,:save,true)
	save && savefig(joinpath(SCRATCH,"Kvphi.eps"),dpi=320)
end

function plot_Kvrθₗ₀(;kwargs...)
	ax1 = subplot(211)
	ax2 = subplot(212)

	plot_Kvrₗ₀(ax=ax1,save=false)
	plot_Kvθₗ₀(ax=ax2,save=false)

	gcf().set_size_inches(8,8)
	tight_layout()
	savefig(joinpath(SCRATCH,"Kvrtheta.eps"),dpi=320)
end

function plot_Kψϕ_s0(;kwargs...)
	kernel_file = joinpath(SCRATCH_kerneldir,"Kψ_imag.fits")
	
	kernel = FITS(kernel_file,"r") do f
				read(f[1])
			end

	s_max = size(kernel,2)

	ax = get(kwargs,:ax) do 
		subplot(111)
	end

	inset_coords = [0.3, 0.65, 0.5, 0.3]
	ax_zoom = ax.inset_axes(inset_coords)

	ls_arr = ["solid","dashed"]
	markers = ["None","."]
	lw = [2,1]
	colors = ["red","blue","green","orange"]

	for s=1:s_max
		for axis in [ax,ax_zoom]
			axis.plot(r_frac,ρ.*kernel[:,s],label="ℓ=$s",
				ls=ls_arr[div(s-1,4)+1],
				color=colors[mod(div(s-1,2),4)+1],
				marker=markers[div(s-1,4)+1],
				markevery=40,linewidth=lw[div(s-1,4)+1])
		end
	end
	ax.set_ylabel(L"\rho\,K_{{ψ_ϕ},ℓ0}",fontsize=14)
	ax.legend(loc="best")
	ax.set_xlim(0.2,r_frac[end])
	ax.set_xlabel(L"$r/R_\odot$",fontsize=12)
	ax.xaxis.set_major_locator(ticker.MaxNLocator(4))

	ax_zoom.set_xlim(0.97,r_frac[end])
	ax_zoom.xaxis.set_major_locator(ticker.MaxNLocator(4))

	ax.indicate_inset_zoom(ax_zoom)
	tight_layout()

	gcf().set_size_inches(8,4)
	save = get(kwargs,:save,false)
	save && savefig(joinpath(SCRATCH,"Kpsiphi.eps"),dpi=320)
end

function plot_Kcₗₘ_reim(;kwargs...)
	lmax = get(kwargs,:lmax,50)
	smax = get(kwargs,:smax,2lmax)
	tmax = get(kwargs,:tmax,smax)

	Kst,header = FITS(joinpath(SCRATCH,"kernels",
		"Kst_δτ_c_lmax$(lmax)_smax$(smax)_tmax$(tmax).fits"),"r") do f
		reinterpret(ComplexF64,read(f[1])),read_header(f[1])
	end

	SHModes = st(0:smax,0:tmax)

	r_frac_cutoff = 0.8
	r_frac_cutoff_zoom = 0.99
	r_frac_trimmed = r_frac[r_frac .> r_frac_cutoff]
	r_frac_zoomed = r_frac[r_frac .> r_frac_cutoff_zoom]
	K_plot = (c.*Kst)[r_frac .> r_frac_cutoff,:]
	K_plot_zoom = (c.*Kst)[r_frac .> r_frac_cutoff_zoom,:]

	modes = [(4,2),(5,3),(7,5)]
	linecolors = ["0.2","0.4","0.6"]
	linestyles = ["solid","dashed","solid"]

	ax_real = get(kwargs,:ax1) do 
		subplot(121)
	end
	ax_imag = get(kwargs,:ax2) do 
		subplot(122)
	end
	inset_coords = [0.15, 0.1, 0.35, 0.35]
	ax_real_zoom = ax_real.inset_axes(inset_coords)
	ax_imag_zoom = ax_imag.inset_axes(inset_coords)

	ymax = 0
	for ((s,t),lc,ls) in zip(modes,linecolors,linestyles)
		SHind = modeindex(SHModes,(s,t))
		if SHind > size(Kst,2)
			continue
		end
		ax_real.plot(r_frac_trimmed,real(K_plot[:,SHind]),
			label="(l=$s,m=$t)",color=lc,ls=ls)
		ax_real_zoom.plot(r_frac_zoomed,real(K_plot_zoom[:,SHind]),color=lc,ls=ls)
		ax_real_zoom.axhline(0,ls="dotted",color="black",zorder=0)
		ymax = max(ymax,maximum(abs,real(K_plot[:,SHind])))
	end
	ax_real.set_xlabel(L"r/R_\odot",fontsize=12)
	ax_real.set_ylabel(L"\Re [cK]",fontsize=12)
	ax_real.xaxis.set_major_locator(ticker.MaxNLocator(4))
	ax_real.legend(loc="best",fontsize=12)
	
	ax_real.set_ylim(-ymax*1.1,ymax*1.1)
	ax_real_zoom.set_ylim(ax_real.get_ylim())
	ax_real_zoom.yaxis.set_major_locator(ticker.NullLocator())
	ax_real.indicate_inset_zoom(ax_real_zoom)
	ax_real.set_title("Real part",fontsize=12)
	
	ymax = 0
	for ((s,t),lc,ls) in zip(modes,linecolors,linestyles)
		SHind = modeindex(SHModes,(s,t))
		if SHind > size(Kst,2)
			continue
		end
		ax_imag.plot(r_frac_trimmed,imag(K_plot[:,SHind]),
			label="(l=$s,m=$t)",color=lc,ls=ls)
		ax_imag_zoom.plot(r_frac_zoomed,imag(K_plot_zoom[:,SHind]),color=lc,ls=ls)
		ax_imag_zoom.axhline(0,ls="dotted",color="black",zorder=0)
		ymax = max(ymax,maximum(abs,imag(K_plot[:,SHind])))
	end
	ax_imag.set_xlabel(L"r/R_\odot",fontsize=12)
	ax_imag.set_ylabel(L"\Im [cK]",fontsize=12)
	ax_imag.xaxis.set_major_locator(ticker.MaxNLocator(4))
	ax_imag.legend(loc="best",fontsize=12)

	ax_imag.set_ylim(-ymax*1.1,ymax*1.1)
	ax_imag_zoom.set_ylim(ax_imag.get_ylim())
	ax_imag_zoom.yaxis.set_major_locator(ticker.NullLocator())
	ax_imag.indicate_inset_zoom(ax_imag_zoom)
	ax_imag.set_title("Imaginary part",fontsize=12)

	tight_layout()
	save = get(kwargs,:save,false)
	save && savefig(joinpath(SCRATCH,"Kst_somemodes.eps"))
end

function plot_Kcₗₘ_spectrum(;kwargs...)
	lmax = get(kwargs,:lmax,50)
	smax = get(kwargs,:smax,2lmax)
	tmax = get(kwargs,:tmax,smax)

	Kst,header = FITS(joinpath(SCRATCH,"kernels",
		"Kst_δτ_c_lmax$(lmax)_smax$(smax)_tmax$(tmax).fits"),"r") do f
		reinterpret(ComplexF64,read(f[1])),read_header(f[1])
	end

	SHModes = st(0:smax,0:tmax)
	ax = get(kwargs,:ax) do 
		subplot(111)
	end

	r₁ = header["R1"]

	K_spectrum = zeros(0:smax,0:tmax)

	r₁_ind = Greenfn_radial.radial_grid_index(r₁)

	for (SHind,(s,t)) in enumerate(SHModes)
		K_spectrum[s,t] = abs2(Kst[r₁_ind,SHind])
	end

	p = ax.imshow(permutedims(K_spectrum),cmap="Greys",origin="lower",
		vmax=maximum(K_spectrum)/10,rasterized=true)
	ax.set_xlabel(L"ℓ",fontsize=12)
	ax.set_ylabel(L"m",fontsize=12)
	ax.set_title("Spectrum",fontsize=12)

	ax.yaxis.set_major_locator(ticker.MaxNLocator(5,integer=true))
	ax.xaxis.set_major_locator(ticker.MaxNLocator(5,integer=true))
end

function plot_Kcₗₘ_reim_spectrum(;kwargs...)

	ax1 = subplot(131)
	ax2 = subplot(132)
	ax3 = subplot(133)

	plot_Kcₗₘ_reim(;kwargs...,ax1=ax1,ax2=ax2)
	plot_Kcₗₘ_spectrum(;kwargs...,ax=ax3)

	gcf().set_size_inches(10,4)
	tight_layout()

	savefig(joinpath(SCRATCH,"Kst_reim_spectrum.eps"),dpi=160)
end

function plot_kernel_timing_scaling_benchmark(n1=Point2D(π/2,0),n2=Point2D(π/2,π/3);
	s_max=10,ℓ_range=20:30:100)

	ns = 3; chunksize = max(1,div(s_max-1,ns))
	s_range = 1:chunksize:s_max
	evaltime = zeros(length(ℓ_range),length(s_range))
	p = Progress(length(evaltime), 1) 
	for (s_ind,s_max) in enumerate(s_range), (ℓind,ℓ) in enumerate(ℓ_range)
		evaltime[ℓind,s_ind] = @elapsed kernel.flow_axisymmetric_without_los(
									n1,n2,s_max,ℓ_range=20:ℓ);
		next!(p)
	end

	subplot(211)
	for (s_ind,s_max) in enumerate(s_range)
		plot(ℓ_range,evaltime[:,s_ind],label="s_max = $s_max",marker="o",ms=4,ls="solid")
	end
	xlabel("Maximum ℓ",fontsize=12)
	ylabel("Runtime [sec]",fontsize=12)
	legend(loc="best",fontsize=12)

	subplot(212)
	plot(s_range,evaltime[end,:],marker="o",ms=4,ls="solid",label="ℓ = $(last(ℓ_range))")
	xlabel("Maximum s",fontsize=12)
	ylabel("Runtime [sec]",fontsize=12)

	tight_layout()
end

function plot_2points_same_longitude(n1,n2,ax;kwargs...)
	θ = kwargs[:θ]
	ϕ = get(kwargs,:ϕ,(n1.ϕ + n2.ϕ)/2)
	r_obs_1 = isa(n1,Point2D) ? r_obs_default/Rsun : n1.r/Rsun
	r_obs_2 = isa(n2,Point2D) ? r_obs_default/Rsun : n2.r/Rsun

	x1 = r_obs_1*sin(n1.ϕ)
	z1 = r_obs_1*cos(n1.ϕ)

	x2 = r_obs_2*sin(n2.ϕ)
	z2 = r_obs_2*cos(n2.ϕ)

	ax.plot([x1],[z1],marker="o",ls="None",ms=6,color="black")
	ax.plot([x2],[z2],marker="o",ls="None",ms=6,color="black")

	xmin = all(x->x>0,cos.(θ)) ? -0.1 : -1.1
	xmax = all(x->x<0,cos.(θ)) ?  0.1 :  1.1
	ymin = all(x->x>0,sin.(θ)) ? -0.1 : -1.1
	ymax = all(x->x<0,sin.(θ)) ?  0.1 :  1.1
	ax.set_xlim(xmin,xmax)
	ax.set_ylim(ymin,ymax)

	ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
	ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
end

function plot_K_longitudinal_slice(n1,n2,K_plot::Matrix{Float64};kwargs...)
	
	θ = get(kwargs,:θ) do 
		LinRange(0,π,size(K_plot,2))
	end
	ϕ = get(kwargs,:ϕ) do 
		(n1.ϕ + n2.ϕ)/2
	end
	dθ = θ[2] - θ[1]

	vmax = maximum(abs.(K_plot))
	vmax_scale = get(kwargs,:vmax_scale,1)
	vmax /= vmax_scale

	fig = get(kwargs,:fig) do
		figure()
	end

	ax = get(kwargs,:ax) do 
		subplot(111,aspect="equal")
	end

	z = r_frac*cos.(θ)'
	x = r_frac*sin.(θ)'

	p = ax.pcolormesh(x,z,K_plot,cmap="RdBu_r",rasterized=true,vmax=vmax,vmin=-vmax)
	
	if n1.ϕ == ϕ && n2.ϕ == ϕ
		plot_2points_same_longitude(n1,n2,ax;kwargs...,θ=θ)
	end

	ax.set_xlim(-0.1,1.1)
	ax.set_ylim(-1.1,1.1)

	ax.set_xlabel(L"r/R_\odot",fontsize=12)
	ax.set_ylabel(L"r/R_\odot",fontsize=12)

	ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
	ax.xaxis.set_major_locator(ticker.MaxNLocator(4))

	divider = axes_grid1.make_axes_locatable(ax)
	cax = divider.new_horizontal(size="5%",pad=0.05)
	fig.add_axes(cax)
	fig.colorbar(p, cax=cax, extend= (vmax_scale==1) ? "neither" : "both")

	mf = ticker.ScalarFormatter(useMathText=true)
	mf.set_powerlimits((-2,2))
	cax.yaxis.set_major_formatter(mf)
	cax.yaxis.set_offset_position("left")
	cax.yaxis.set_major_locator(ticker.MaxNLocator(3))
end

function plot_Kc_longitudinal_slice(K_fits_filename;kwargs...)
	K,header = FITS(K_fits_filename,"r") do f
		c.*read(f[1]),read_header(f[1])
	end
	
	x1 = Point3D(header["R1"],header["TH1"],header["PHI1"])
	x2 = Point3D(header["R2"],header["TH2"],header["PHI2"])

	θ_slice = LinRange(header["THSLMN"],header["THSLMX"],header["NTH"])
	ϕ_slice = header["PHSL"]

	plot_K_longitudinal_slice(x1,x2,K;θ=θ_slice,ϕ=ϕ_slice,kwargs...)
end

function plot_Kc_3D_longitudinal_slice_twopanels(;kwargs...)
	fig = get(kwargs,:fig) do 
		figure()
	end
	ax1 = get(kwargs,:ax_long_1) do 
		subplot(121,aspect="equal")
	end
	ax2 = get(kwargs,:ax_long_2) do 
		subplot(122,aspect="equal")
	end

	filename = joinpath(SCRATCH_kerneldir,"Krθ_from_Kst.fits")
	plot_Kc_longitudinal_slice(filename;kwargs...,ax=ax1,fig = fig)

	header = FITS(filename,"r") do f
		read_header(f[1])
	end

	lmax = header["LMAX"]

	filename = joinpath(SCRATCH_kerneldir,
			"Kc_3D_long_slice_jcutoff$(lmax).fits")

	plot_Kc_longitudinal_slice(filename;kwargs...,ax=ax2,fig = fig)

	ax1.set_title("This work",fontsize=12)
	ax2.set_title("3D kernel",fontsize=12)

	save = get(kwargs,:save,true)
	if save
		fig.set_size_inches(8,4)
		tight_layout()
		fig.savefig(joinpath(SCRATCH,"compare_3D_kernel_longslice.eps"),
		dpi=get(kwargs,:dpi,160))
	end
end

function plot_Ku_longitudinal_slice(;kwargs...)

	K_fits_filename = joinpath(SCRATCH_kerneldir,"Ku_rθslice_from_Kₗₘ.fits")

	K,header = FITS(K_fits_filename,"r") do f
		c.*read(f[1]),read_header(f[1])
	end
	
	x1 = Point3D(header["R1"],header["TH1"],header["PHI1"])
	x2 = Point3D(header["R2"],header["TH2"],header["PHI2"])

	θ_slice = LinRange(header["THSLMN"],header["THSLMX"],header["NTH"])
	ϕ = header["PHSL"]

	fig = figure()
	
	ax1 = subplot(131,aspect="equal")
	vr_scale = get(kwargs,:vr_scale,1)
	plot_K_longitudinal_slice(x1,x2,K[:,:,1];θ=θ_slice,ϕ=ϕ,kwargs...,
		fig=fig,ax=ax1,vmax_scale=vr_scale)

	ax2 = subplot(132,aspect="equal")
	vθ_scale = get(kwargs,:vθ_scale,1)
	plot_K_longitudinal_slice(x1,x2,K[:,:,2];θ=θ_slice,ϕ=ϕ,kwargs...,
		fig=fig,ax=ax2,vmax_scale=vθ_scale)

	ax3 = subplot(133,aspect="equal")
	vϕ_scale = get(kwargs,:vϕ_scale,1)
	plot_K_longitudinal_slice(x1,x2,K[:,:,3];θ=θ_slice,ϕ=ϕ,kwargs...,
		fig=fig,ax=ax3,vmax_scale=vϕ_scale)

	fig.set_size_inches(12,4)
	fig.tight_layout()
end

function plot_2points_same_latitude(n1,n2,ax;kwargs...)
	θ = get(kwargs,:θ,(n1.θ + n2.θ)/2)
	ϕ = kwargs[:ϕ]
	r_obs_1 = isa(n1,Point2D) ? r_obs_default/Rsun : n1.r/Rsun
	r_obs_2 = isa(n2,Point2D) ? r_obs_default/Rsun : n2.r/Rsun

	x1 = r_obs_1*cos(n1.ϕ)
	y1 = r_obs_1*sin(n1.ϕ)

	x2 = r_obs_2*cos(n2.ϕ)
	y2 = r_obs_2*sin(n2.ϕ)

	ax.plot([x1],[y1],marker="o",ls="None",ms=6,mfc="yellow",mec="black")
	ax.plot([x2],[y2],marker="o",ls="None",ms=6,mfc="yellow",mec="black")

	xmin = all(x->x>0,cos.(ϕ)) ? -0.1 : minimum(cos.(ϕ))*1.1
	xmax = all(x->x<0,cos.(ϕ)) ?  0.1 : maximum(cos.(ϕ))*1.1
	ymin = all(x->x>0,sin.(ϕ)) ? -0.1 : minimum(sin.(ϕ))*1.1
	ymax = all(x->x<0,sin.(ϕ)) ?  0.1 : maximum(sin.(ϕ))*1.1

	ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
	ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
end

function plot_K_latitudinal_slice(n1,n2,K_plot::Matrix{Float64};kwargs...)
	
	ϕ = get(kwargs,:ϕ) do 
		LinRange(0,2π,4size(K_plot,2))
	end
	θ = get(kwargs,:θ) do 
		(n1.θ + n2.θ)/2
	end

	vmax = maximum(abs.(K_plot))
	vmax_scale = get(kwargs,:vmax_scale,1)
	vmax /= vmax_scale

	fig = get(kwargs,:fig) do
		figure()
	end

	ax = get(kwargs,:ax) do 
		subplot(111,aspect="equal")
	end

	x = r_frac*cos.(ϕ)';
	y = r_frac*sin.(ϕ)';

	p = ax.pcolormesh(x,y,K_plot,
		cmap="RdBu_r",rasterized=true,vmax=vmax,vmin=-vmax)

	if n1.θ == θ && n2.θ == θ
		plot_2points_same_latitude(n1,n2,ax;kwargs...,ϕ=ϕ)
	end

	ax.set_xlabel(L"r/R_\odot",fontsize=12)
	ax.set_ylabel(L"r/R_\odot",fontsize=12)

	ax.set_xlim(-1.1,1.1)
	ax.set_ylim(-1.1,1.1)

	ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
	ax.xaxis.set_major_locator(ticker.MaxNLocator(4))

	divider = axes_grid1.make_axes_locatable(ax)
	cax = divider.new_horizontal(size="5%",pad=0.05)
	fig.add_axes(cax)
	fig.colorbar(p, cax=cax, extend= (vmax_scale==1) ? "neither" : "both")

	mf = ticker.ScalarFormatter(useMathText=true)
	mf.set_powerlimits((-2,2))
	cax.yaxis.set_major_formatter(mf)
	cax.yaxis.set_offset_position("left")
	cax.yaxis.set_major_locator(ticker.MaxNLocator(3))
end

function plot_Kc_latitudinal_slice(K_fits_filename;kwargs...)
	K,header = FITS(K_fits_filename,"r") do f
		c.*read(f[1]),read_header(f[1])
	end
	
	x1 = Point3D(header["R1"],header["TH1"],header["PHI1"])
	x2 = Point3D(header["R2"],header["TH2"],header["PHI2"])

	ϕ = LinRange(header["PHSLMN"],header["PHSLMX"],header["NPHI"])

	plot_K_latitudinal_slice(x1,x2,K;ϕ=ϕ,kwargs...)
end

function plot_Kc_3D_latitudinal_slice_twopanels(;kwargs...)
	fig = get(kwargs,:fig) do 
		figure()
	end
	ax1 = get(kwargs,:ax_lat_1) do
		subplot(121,aspect="equal")
	end
	ax2 = get(kwargs,:ax_lat_2) do
		subplot(122,aspect="equal")
	end

	filename = joinpath(SCRATCH_kerneldir,"Krϕ_from_Kst.fits")
	plot_Kc_latitudinal_slice(filename;kwargs...,ax=ax1,fig=fig)

	header = FITS(filename,"r") do f
		read_header(f[1])
	end

	lmax = header["LMAX"]

	filename = joinpath(SCRATCH_kerneldir,
		"Kc_3D_lat_slice_jcutoff$(lmax).fits")

	plot_Kc_latitudinal_slice(filename;kwargs...,ax=ax2,fig=fig)

	ax1.set_title("This work",fontsize=12)
	ax2.set_title("3D kernel",fontsize=12)

	save = get(kwargs,:save,true)
	if save 
		fig.set_size_inches(8,4)
		tight_layout()
		fig.savefig(joinpath(SCRATCH,"compare_3D_kernel_latslice.eps"),
		dpi=get(kwargs,:dpi,160))
	end
end

function plot_Ku_latitudinal_slice(;kwargs...)
	
	K_fits_filename = joinpath(SCRATCH_kerneldir,"Ku_rϕslice_from_Kₗₘ.fits")

	K,header = FITS(K_fits_filename,"r") do f
		c.*read(f[1]),read_header(f[1])
	end
	
	x1 = Point3D(header["R1"],header["TH1"],header["PHI1"])
	x2 = Point3D(header["R2"],header["TH2"],header["PHI2"])

	θ = header["THSL"]
	ϕ_slice = LinRange(header["PHSLMN"],header["PHSLMX"],header["NPHI"])

	fig = figure()
	
	ax1 = subplot(131,aspect="equal")
	vr_scale = get(kwargs,:vr_scale,1)
	plot_K_latitudinal_slice(x1,x2,K[:,:,1];θ=θ,ϕ=ϕ_slice,fig=fig,ax=ax1,
		kwargs...,vmax_scale=vr_scale)

	ax2 = subplot(132,aspect="equal")
	vθ_scale = get(kwargs,:vθ_scale,1)
	plot_K_latitudinal_slice(x1,x2,K[:,:,2];θ=θ,ϕ=ϕ_slice,fig=fig,ax=ax2,
		kwargs...,vmax_scale=vθ_scale)

	ax3 = subplot(133,aspect="equal")
	vϕ_scale = get(kwargs,:vϕ_scale,1)
	plot_K_latitudinal_slice(x1,x2,K[:,:,3];θ=θ,ϕ=ϕ_slice,fig=fig,ax=ax3,
		kwargs...,vmax_scale=vϕ_scale)

	fig.set_size_inches(12,4)
	fig.tight_layout()
end

function plot_Kc_3D_lat_long_slice_twopanels(;kwargs...)
	fig = figure()
	ax_lat_1 = subplot(221,aspect="equal")
	ax_lat_2 = subplot(222,aspect="equal")
	ax_long_1 = subplot(223,aspect="equal")
	ax_long_2 = subplot(224,aspect="equal")

	vmax_scale_lat = get(kwargs,:vmax_scale_lat) do 
		200
	end
	vmax_scale_long = get(kwargs,:vmax_scale_long) do 
		20
	end

	plot_Kc_3D_latitudinal_slice_twopanels(fig=fig,ax_lat_1=ax_lat_1,
		ax_lat_2=ax_lat_2,vmax_scale=vmax_scale_lat,save=false)
	plot_Kc_3D_longitudinal_slice_twopanels(fig=fig,ax_long_1=ax_long_1,
		ax_long_2=ax_long_2,vmax_scale=vmax_scale_long,save=false)

	fig.set_size_inches(8,8)
	tight_layout()

	fig.savefig(joinpath(SCRATCH,"compare_3D_kernel_latlongslice.eps"),
		dpi=get(kwargs,:dpi,160))
end

function plot_Kv_10_different_n1n2()
	filename = joinpath(SCRATCH_kerneldir,"K_δτ_v10_equator.fits")
	K1 = FITS(filename,"r") do f
		read(f[1])
	end

	filename = joinpath(SCRATCH_kerneldir,"K_δτ_v10_diagonal.fits")
	K2 = FITS(filename,"r") do f
		read(f[1])
	end

	filename = joinpath(SCRATCH_kerneldir,"K_δτ_v10_meridian.fits")
	K3 = FITS(filename,"r") do f
		read(f[1])
	end

	scale = max(maximum(abs,K1),maximum(abs,K2),maximum(abs,K3))

	ax1 = subplot(131)
	ax2 = subplot(132)
	ax3 = subplot(133)

	ax1.plot(r_frac,K1./scale,color="black",lw=2)
	ax2.plot(r_frac,K2./scale,color="black",lw=2)
	ax3.plot(r_frac,K3./scale,color="black",lw=2)

	ax2.set_ylim(ax1.get_ylim())
	ax3.set_ylim(ax1.get_ylim())

	for ax in [ax1,ax2,ax3]
		ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
		ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
		ax.set_xlabel(L"$R/R_\odot$",fontsize=12)
	end

	ax1.set_ylabel("Kvϕ_10",fontsize=12)

	gcf().set_size_inches(8,2)
	tight_layout()
	savefig(joinpath(SCRATCH,"kernel_pairs_of_points.eps"),dpi=320)
end


function plot_kernel_with_and_without_los_sameheight(args...;kwargs...)

	K2D = kernel.flow_axisymmetric_without_los(args...;kwargs...)
	Base.GC.gc()
	K2D_los = kernel.flow_axisymmetric_with_los(args...;kwargs...)

	plot(r_frac,K2D[:,-1,1],label="without LoS")
	plot(r_frac,K2D_los[:,-1,1],label="with LoS")
	xlim(0.7,r_frac[end])
	legend(loc="best",fontsize=12)
	xlabel(L"$r/R_\odot$",fontsize=12)
	ylabel("Kernel [s/(cm/s Mm)]",fontsize=12)
	title("ϕ component of kernel")
	tight_layout()
end

# Green function

# function plot_Gfn_with_krishnendu(ℓ=40,ν=3e-3;δc=0,kwargs...)
# 	# Gfn_krishnendu_path = "/scratch/jb6888/Greenfn_krishnendu"

# 	r_src = get(kwargs,:r_src,r_src_default)

# 	c_scale=1+δc

# 	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(r_src,c_scale=c_scale)
# 	println("Loading Greenfn from $Gfn_path_src")
# 	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr

# 	ν_ind = argmin(abs.(ν_arr .- ν))

# 	ν_ongrid = ν_arr[ν_ind]
# 	ω = 2π*ν_ongrid
# 	γ = Greenfn_radial.γ_damping
# 	ω_γ = ω-im*γ

# 	ρ = Greenfn_radial.ρ
# 	c = Greenfn_radial.c
# 	g = Greenfn_radial.g
# 	N² = Greenfn_radial.N2

# 	@. N²+= g^2/c^2*(1-1/c_scale^2)
# 	@. c*=c_scale

# 	a = @. g/c^2
# 	Sₗ² = @. ℓ*(ℓ+1)*c^2/r^2
#     b = @. (Sₗ² /ω_γ^2 -1)/ (ρ*c^2)
#     c = @. ρ*(ω_γ^2 - N²)

#     matrix_kri = ho_endpt_so.genmatrix(a,b,c,r)
#     src = Greenfn_radial.source(ω,ℓ,r_src=r_src)[1]
#     src_kri = vcat(src[nr:end],[0,0],src[1:nr-1])
#     sol_kri = matrix_kri\src_kri
#     ξ_kri = sol_kri[1:nr]
# 	p_kri = sol_kri[nr+1:2nr]

# 	G1_kri = @. Greenfn_radial.Ω(ℓ,0)/(ρ*r*ω^2) * p_kri
	
# 	# Gfn_krishnendu = npzread(joinpath(Gfn_krishnendu_path,
# 	# 	"Gr_nuind_$(ν_ind-1)_ell$(ℓ)_dc$(@sprintf "%g" δc).npy"))[1,:]
# 	# r_krishnendu = readdlm("krishnendu_solar_model/model0.2to1")[:,1][end:-1:1]
	
# 	@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs

# 	proc_id = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,(ℓ,ν_ind),num_procs)
# 	Gfn_fits = Greenfn_radial.Gfn_fits_files(Gfn_path_src,(proc_id,))
# 	G0_jishnu = Greenfn_radial.read_Gfn_file_at_index(Gfn_fits,
# 	   ℓ_arr,1:Nν_Gfn,(ℓ,ν_ind),num_procs,:,1,1,1)
# 	G1_jishnu = Greenfn_radial.read_Gfn_file_at_index(Gfn_fits,
# 	   ℓ_arr,1:Nν_Gfn,(ℓ,ν_ind),num_procs,:,2,1,1)

# 	# display(hcat(ξ_kri,G0_jishnu))

# 	ax1=subplot(221)
# 	plot(r_frac,real(ξ_kri),label="Mandal et al. [2017]",color="grey")
# 	plot(r_frac,real(G0_jishnu),markevery=20,marker=".",
# 		ls="None",label="This work",color="black")
# 	xlim(0.995,r_frac[end]);
# 	ylabel(L"\Re[G^{(-1)}_{(-1),\ell\omega}]",fontsize=12)
# 	legend(loc="best")
# 	xlabel(L"r/R_\odot",fontsize=12)
# 	ax1.xaxis.set_major_locator(ticker.MaxNLocator(3))

# 	ax2=subplot(222)
# 	plot(r_frac,imag(ξ_kri),label="Mandal et al. [2017]",color="grey")
# 	plot(r_frac,imag(G0_jishnu),markevery=20,marker=".",
# 		ls="None",label="This work",color="black")
# 	xlim(0.995,r_frac[end]);
# 	legend(loc="best")
# 	ax2.xaxis.set_major_locator(ticker.MaxNLocator(3))
# 	xlabel(L"r/R_\odot",fontsize=12)
# 	ylabel(L"\Im[G^{(-1)}_{(-1),\ell\omega}]",fontsize=12)

# 	ax3=subplot(223)
# 	plot(r_frac,real(G1_kri),label="Mandal et al. [2017]",color="grey")
# 	plot(r_frac,real(G1_jishnu),markevery=20,marker=".",
# 		ls="None",label="This work",color="black")
# 	xlim(0.995,r_frac[end]);
# 	ylabel(L"\Re[G^{(+1)}_{(-1),\ell\omega}]",fontsize=12)
# 	legend(loc="best")
# 	xlabel(L"r/R_\odot",fontsize=12)
# 	ax3.xaxis.set_major_locator(ticker.MaxNLocator(3))

# 	ax4=subplot(224)
# 	plot(r_frac,imag(G1_kri),label="Mandal et al. [2017]",color="grey")
# 	plot(r_frac,imag(G1_jishnu),markevery=20,marker=".",
# 		ls="None",label="This work",color="black")
# 	xlim(0.995,r_frac[end]);
# 	legend(loc="best")
# 	xlabel(L"r/R_\odot",fontsize=12)
# 	ylabel(L"\Im[G^{(+1)}_{(-1),\ell\omega}]",fontsize=12)
# 	ax4.xaxis.set_major_locator(ticker.MaxNLocator(3))

# 	tight_layout()
# end

function plot_reciprocity(G_reciprocity;kwargs...)

	Gfn_path_src = Greenfn_radial.Gfn_path_from_source_radius(
					get(kwargs,:r_src,r_src_default))
	
	@load(joinpath(Gfn_path_src,"parameters.jld2"),
		ν_arr,Nν_Gfn,ℓ_arr,num_procs)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
	ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)

	vmax = maximum(G_reciprocity)
	norm= colors.LogNorm(vmin=vmax/1e4, vmax=vmax)

	subplot(131)
	pcolormesh(ℓ_range,ν_arr[ν_ind_range].*1e3,
		G_reciprocity[:,:,1],cmap="Greys",rasterized=true,norm=norm)
	title(L"|G^1_0\,(r_2,r_1)|",fontsize=12)
	xlabel("ℓ",fontsize=12)
	ylabel("Frequency [mHz]",fontsize=12)
	gca().xaxis.set_major_locator(ticker.MaxNLocator(4,integer=true))
	gca().yaxis.set_major_locator(ticker.MaxNLocator(5))
	colorbar(extend="min")
	
	subplot(132)
	pcolormesh(ℓ_range,ν_arr[ν_ind_range].*1e3,
		G_reciprocity[:,:,2],cmap="Greys",rasterized=true,norm=norm)
	title(L"|G^0_1\,(r_1,r_2)|",fontsize=12)
	xlabel("ℓ",fontsize=12)
	gca().xaxis.set_major_locator(ticker.MaxNLocator(4,integer=true))
	gca().yaxis.set_major_locator(ticker.MaxNLocator(5))
	colorbar(extend="min")

	subplot(133)
	Gdiff = G_reciprocity[:,:,1] .- G_reciprocity[:,:,2]
	vmax = maximum(abs.(Gdiff))
	norm= colors.LogNorm(vmin=vmax/1e4, vmax=vmax)
	pcolormesh(ℓ_range,ν_arr[ν_ind_range].*1e3,
		Gdiff,cmap="Greys",rasterized=true,norm=norm)
	title("Difference",fontsize=12)
	xlabel("ℓ",fontsize=12)
	gca().xaxis.set_major_locator(ticker.MaxNLocator(4,integer=true))
	gca().yaxis.set_major_locator(ticker.MaxNLocator(5))
	colorbar(extend="min")	

	tight_layout()
end

function plot_reciprocity(;kwargs...)
	G_reciprocity = Greenfn_radial.Gfn_reciprocity(;kwargs...)
	plot_reciprocity(G_reciprocity;kwargs...)
end

function plot_solar_damping(;fit_order=5)
	HMIfreq=readdlm("m181q.1216");
	ℓ_HMI,ν_HMI,γ_HMI,Δγ_HMI = HMIfreq[:,1],HMIfreq[:,3],HMIfreq[:,5],HMIfreq[:,10];

	# Fit modes above ℓ=11 and 2mHz<ν<4mHz 
	mode_filter = (ℓ_HMI.>11) .& (ν_HMI .> 2e3) .& (ν_HMI .< 4.5e3);
	
	ν_HMI = ν_HMI[mode_filter];
	ℓ_HMI = ℓ_HMI[mode_filter];
	γ_HMI = γ_HMI[mode_filter];
	Δγ_HMI = Δγ_HMI[mode_filter];

	# Fit γ(ω) in Hz, the HMI frequencies are in μHz
	γ_damping = polyfit(ν_HMI.*(2π*1e-6),γ_HMI.*(2π*1e-6),fit_order);

	γ_fit = @. γ_damping(ν_HMI.*(2π*1e-6))*1e6/2π;

	ν_HMI_mHz = ν_HMI./1e3;

	ax_ℓν = subplot2grid((3,3),(1,0),colspan=2,rowspan=2)
	norm = colors.LogNorm(vmin=minimum(γ_HMI),vmax=maximum(γ_HMI))
	scatter(ℓ_HMI,ν_HMI_mHz,c=γ_HMI,cmap="Greys",s=10,norm=norm)
	sm = cm.ScalarMappable(norm=norm, cmap="Greys")
	sm.set_array(γ_HMI)
	colorbar(sm, ax=ax_ℓν,format=ticker.LogFormatter(base=10,
						labelOnlyBase=false, minor_thresholds=(10,2)),
						orientation="horizontal")
	xlabel("ℓ",fontsize=14)
	ylabel("Frequency [mHz]",fontsize=14)

	ax_ℓ = subplot2grid((3,3),(0,0),colspan=2,rowspan=1)
	plot(ℓ_HMI,γ_HMI,"o",ls="None",ms=4,mfc="0.7",mec="0.3")
	ylabel(L"$\gamma$ [$\mu$Hz]",fontsize=14);

	ax_ν = subplot2grid((3,3),(1,2),rowspan=2,colspan=1)
	plot(γ_HMI,ν_HMI_mHz,"o",ls="None",ms=4,mfc="0.3",mec="0.6")

	p = sortperm(ν_HMI_mHz)
	ν_HMI_mHz = ν_HMI_mHz[p]
	γ_fit = γ_fit[p]

	suffix = (fit_order==1) ? "st" : (fit_order==2) ? "nd" : 
			(fit_order==3) ? "rd" : "th"

	plot(γ_fit,ν_HMI_mHz,color="black",label="fit")
	legend(loc="lower right",fontsize=16)

	xlabel(L"$\gamma$ [$\mu$Hz]",fontsize=16);
	cb_dummy=colorbar(sm, ax=ax_ν,orientation="horizontal")
	cb_dummy.ax.set_visible(false)

	ax_ℓν.yaxis.set_major_locator(ticker.MaxNLocator(4))
	ax_ℓν.xaxis.set_major_locator(ticker.MaxNLocator(4))

	ax_ν.yaxis.set_major_locator(ticker.NullLocator())
	ax_ℓ.xaxis.set_major_locator(ticker.NullLocator())

	gcf().set_size_inches(8,6)
	tight_layout()
	gcf().subplots_adjust(hspace=0,wspace=0)


	savefig(joinpath(SCRATCH,"solar_damping.eps"),dpi=160)

	γ_damping
end

function plot_δG_isotropicδc(xobs,xsrc;kwargs...)
	Gfn_path_src = Greenfn_3D.Gfn_path_from_source_radius(xsrc)
	@load(joinpath(Gfn_path_src,"parameters.jld2"),ν_start_zeros,Nν_Gfn,ν_arr,ℓ_arr)

	δGFB = Greenfn_3D.δGrr_isotropicδc_firstborn_integrated_over_angle(xobs,xsrc;kwargs...)
	δGfull = Greenfn_3D.δGrr_isotropicδc_GminusG0(xobs,xsrc;kwargs...)

	ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)

	subplot(211)
	plot(ν_arr.*1e3,real(δGFB[ν_start_zeros .+ (1:Nν_Gfn)]),
		label="First Born",color="grey")
	plot(ν_arr.*1e3,real(δGfull[ν_start_zeros .+ (1:Nν_Gfn)]),
		marker=".",ls="None",label="G-G₀",color="black")
	title("Change in Gᵣᵣ for ℓ=$(length(ℓ_range)==1 ? first(ℓ_range) : ℓ_range)",
		fontsize=15)
	legend(loc="best",fontsize=12)
	ylabel(L"\Re \left[\delta G_{rr} \right]",fontsize=15)
	gca().xaxis.set_major_locator(ticker.MaxNLocator(5))
	gca().yaxis.set_major_locator(ticker.MaxNLocator(4))

	subplot(212)
	plot(ν_arr.*1e3,imag(δGFB[ν_start_zeros .+ (1:Nν_Gfn)]),
		label="First Born",color="grey")
	plot(ν_arr.*1e3,imag(δGfull[ν_start_zeros .+ (1:Nν_Gfn)]),
		marker=".",ls="None",label="G-G₀",color="black")
	xlabel("Frequency [mHz]",fontsize=15)
	legend(loc="best",fontsize=12)
	ylabel(L"\Im \left[\delta G_{rr} \right]",fontsize=15)
	gca().xaxis.set_major_locator(ticker.MaxNLocator(5))
	gca().yaxis.set_major_locator(ticker.MaxNLocator(4))

	tight_layout()

	savefig(joinpath(SCRATCH,"dGrr.eps"))
end

end