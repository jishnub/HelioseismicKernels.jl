using HelioseismicKernels
using Test

n1,n2=Point2D(π/3,π/4),Point2D(2π/3,π/3);
r_src=Greenfn_radial.r_src_default;
r_obs=Greenfn_radial.r_obs_default;
xsrc1,xsrc2,xobs1,xobs2=Point3D(r_src,n1),Point3D(r_src,n2),Point3D(r_obs,n1),Point3D(r_obs,n2);

@testset "kernel components" begin
	@testset "soundspeed" begin
	    function uniformvalidation(::Kernel.soundspeed,m::Kernel.SeismicMeasurement,xobs1,xobs2,
	    	los::Directions.los_direction=Directions.los_radial();SHModes=LM(0:0,0:0))

			ℓ_range = 1:1

			hω = Crosscov.hω(m,xobs1,xobs2,los;ℓ_range=ℓ_range)
			
			Kδcₛₜ = Kernel.kernel_δcₛₜ(m,xobs1,xobs2,los;ℓ_range=ℓ_range,SHModes=SHModes,hω=hω,save=false);
			
			Kδc₀₀ = Kernel.kernel_δc₀₀(m,xobs1,xobs2,los;ℓ_range=ℓ_range,hω=hω,save=false);
			
			@test Kδcₛₜ[:,(0,0)]≈Kδc₀₀
		end

		@testset "TravelTimes" begin
			@testset "radial" begin
				uniformvalidation(Kernel.soundspeed(),Kernel.TravelTimes(),xobs1,xobs2,Directions.los_radial())
				uniformvalidation(Kernel.soundspeed(),Kernel.TravelTimes(),xobs1,xobs2,Directions.los_radial(),
					SHModes=LM(0:2,0:2))
			end
			@testset "line-of-sight" begin
				uniformvalidation(Kernel.soundspeed(),Kernel.TravelTimes(),xobs1,xobs2,Directions.los_earth())
				uniformvalidation(Kernel.soundspeed(),Kernel.TravelTimes(),xobs1,xobs2,Directions.los_earth(),
					SHModes=LM(0:2,0:2))
			end
		end
		@testset "Amplitudes" begin
			@testset "radial" begin
				uniformvalidation(Kernel.soundspeed(),Kernel.Amplitudes(),xobs1,xobs2,Directions.los_radial())
				uniformvalidation(Kernel.soundspeed(),Kernel.Amplitudes(),xobs1,xobs2,Directions.los_radial(),
					SHModes=LM(0:2,0:2))
			end
			@testset "line-of-sight" begin
				uniformvalidation(Kernel.soundspeed(),Kernel.Amplitudes(),xobs1,xobs2,Directions.los_earth())
				uniformvalidation(Kernel.soundspeed(),Kernel.Amplitudes(),xobs1,xobs2,Directions.los_earth(),
					SHModes=LM(0:2,0:2))
			end
		end
	end

	@testset "flows" begin
		function uniformvalidation(::Kernel.flows,m::Kernel.SeismicMeasurement,xobs1,xobs2,
			los::Directions.los_direction=Directions.los_radial();SHModes=LM(1:1,0:0))

			ℓ_range = 1:1

			hω = Crosscov.hω(m,xobs1,xobs2,los;ℓ_range=ℓ_range)
			
			Krθϕₛ₀ = Kernel.kernel_uₛ₀_rθϕ(m,xobs1,xobs2,los;hω=hω,SHModes=SHModes,ℓ_range=ℓ_range,save=false)
			Krθϕₛ₀_2 = Kernel.kernel_uₛ₀_rθϕ_2(m,xobs1,xobs2,los;hω=hω,SHModes=SHModes,ℓ_range=ℓ_range,save=false)
			
			Kℑu⁺₁₀ = Kernel.kernel_ℑu⁺₁₀(m,xobs1,xobs2,los;ℓ_range=ℓ_range,hω=hω,save=false);
			@testset "Ks0 from Kst" begin
				@test @view(Krθϕₛ₀[:,3,(1,0)]) ≈ Kℑu⁺₁₀
			end			
			@testset "Ks0 directly" begin
				@test @view(Krθϕₛ₀_2[:,3,(1,0)]) ≈ Kℑu⁺₁₀
			end
		end

		@testset "TravelTimes" begin
			@testset "radial" begin
				uniformvalidation(Kernel.flows(),Kernel.TravelTimes(),xobs1,xobs2,Directions.los_radial())
				uniformvalidation(Kernel.flows(),Kernel.TravelTimes(),xobs1,xobs2,Directions.los_radial(),
					SHModes=LM(1:2,0:2))
			end
			@testset "line-of-sight" begin
				uniformvalidation(Kernel.flows(),Kernel.TravelTimes(),xobs1,xobs2,Directions.los_earth())
				uniformvalidation(Kernel.flows(),Kernel.TravelTimes(),xobs1,xobs2,Directions.los_earth(),
					SHModes=LM(1:2,0:2))
			end
		end

		@testset "Amplitudes" begin
			@testset "radial" begin
				uniformvalidation(Kernel.flows(),Kernel.Amplitudes(),xobs1,xobs2,Directions.los_radial())
				uniformvalidation(Kernel.flows(),Kernel.Amplitudes(),xobs1,xobs2,Directions.los_radial(),
					SHModes=LM(1:2,0:2))
			end
			@testset "line-of-sight" begin
				uniformvalidation(Kernel.flows(),Kernel.Amplitudes(),xobs1,xobs2,Directions.los_earth())
				uniformvalidation(Kernel.flows(),Kernel.Amplitudes(),xobs1,xobs2,Directions.los_earth(),
					SHModes=LM(1:2,0:2))
			end
		end
	end
end;

@testset "kernel PB Hansen" begin
    @testset "Equator" begin
    	n1,n2 = Point2D(π/2,0), Point2D(π/2,π/3)
        n1eq,n2eq = Point2D(Equator(),n1.ϕ), Point2D(Equator(),n2.ϕ)

        ℓ_range = 20:21
        SHModes = LM(0:1,0:1)
        @testset "soundspeed" begin
            Kpiby2=Kernel.kernel_δcₛₜ(Kernel.TravelTimes(),
            	n1,n2,Directions.los_earth(),
            	ℓ_range=ℓ_range,SHModes=SHModes,print_timings=false);
            
            Kequator=Kernel.kernel_δcₛₜ(Kernel.TravelTimes(),
            	n1eq,n2eq,Directions.los_earth(),
            	ℓ_range=ℓ_range,SHModes=SHModes,print_timings=false);

            @test Kequator ≈ Kpiby2
        end
        @testset "flows" begin
            Kpiby2=Kernel.kernel_uₛₜ(Kernel.TravelTimes(),
            	n1,n2,Directions.los_earth(),
            	ℓ_range=ℓ_range,SHModes=SHModes,print_timings=false);
            
            Kequator=Kernel.kernel_uₛₜ(Kernel.TravelTimes(),
            	n1eq,n2eq,Directions.los_earth(),
            	ℓ_range=ℓ_range,SHModes=SHModes,print_timings=false);
            
            @test Kequator ≈ Kpiby2
        end
    end;
end;

@testset "Ks0 computed directly and from Kst" begin

	function test(m,los;SHModes=LM(1:1,0:0),ℓ_range = 2:2)
		hω = Crosscov.hω(m,xobs1,xobs2,los;ℓ_range=ℓ_range)

		Krθϕₛ₀ = Kernel.kernel_uₛ₀_rθϕ(m,xobs1,xobs2,los;hω=hω,
					SHModes=SHModes,ℓ_range=ℓ_range,save=false)
		Krθϕₛ₀_2 = Kernel.kernel_uₛ₀_rθϕ_2(m,xobs1,xobs2,los;hω=hω,
					SHModes=SHModes,ℓ_range=ℓ_range,save=false)

		@test Krθϕₛ₀ ≈ Krθϕₛ₀_2
	end

	test(Kernel.TravelTimes(),Directions.los_radial())
	test(Kernel.TravelTimes(),Directions.los_radial(),SHModes=LM(0:4,0:4))
	test(Kernel.TravelTimes(),Directions.los_earth())
	test(Kernel.TravelTimes(),Directions.los_earth(),SHModes=LM(0:4,0:4))
end;

@testset "multiple points" begin

	function testmultiplepoints(m,los)
		ℓ_range = 1:1
		hω = Crosscov.hω(m,xobs1,xobs2,los;ℓ_range=ℓ_range)
	    K=Kernel.kernel_ℑu⁺₁₀(m,n1,n2,los,ℓ_range=ℓ_range,hω=hω,
	    	ν_ind_range=axes(hω,1));
	    K2=Kernel.kernel_ℑu⁺₁₀(m,n1,[n2],los,ℓ_range=ℓ_range,hω=hω,
	    	ν_ind_range=axes(hω,1));
	    @test K ≈ K2
	end

	@testset "TravelTimes" begin
		@testset "radial" begin
			testmultiplepoints(Kernel.TravelTimes(),Directions.los_radial())
		end
		@testset "line-of-sight" begin
			testmultiplepoints(Kernel.TravelTimes(),Directions.los_earth())
		end
	end
	@testset "Amplitudes" begin
		@testset "radial" begin
			testmultiplepoints(Kernel.Amplitudes(),Directions.los_radial())
		end
		@testset "line-of-sight" begin
			testmultiplepoints(Kernel.Amplitudes(),Directions.los_earth())
		end
	end
end;
