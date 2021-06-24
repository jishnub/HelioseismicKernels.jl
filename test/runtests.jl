using Test
using SphericalHarmonicModes

n1, n2 = HelioseismicKernels.Point2D(0, 0), HelioseismicKernels.Point2D(pi/4, 0);
n1′, n2′ = HelioseismicKernels.Point2D(pi/2, 0), HelioseismicKernels.Point2D(pi/2, pi/4);
r_src = HelioseismicKernels.r_src_default;
r_obs = HelioseismicKernels.r_obs_default;
xsrc1 = HelioseismicKernels.Point3D(r_src, n1);
xsrc2 = HelioseismicKernels.Point3D(r_src, n2);
xobs1 = HelioseismicKernels.Point3D(r_obs, n1);
xobs2 = HelioseismicKernels.Point3D(r_obs, n2);

@testset "all tests" begin
	@testset "rotation of Y₀₀" begin
		Y1′2′ = HelioseismicKernels.los_projected_biposh_spheroidal(HelioseismicKernels.computeY₀₀, n1′, n2′, HelioseismicKernels.los_earth(), 1:10)
		Y1′2′_12 = HelioseismicKernels.los_projected_biposh_spheroidal_Y₀₀(n1, n2, n1′, n2′, HelioseismicKernels.los_earth(), 1:10)
		@test Y1′2′ ≈  Y1′2′_12
	end
	@testset "rotated cross covariances" begin
		C = HelioseismicKernels.Cω(nothing, (n1, n1′), (n2, n2′), HelioseismicKernels.los_earth());
		C1 = HelioseismicKernels.Cω(nothing, n1, n2, HelioseismicKernels.los_earth());
		C2 = HelioseismicKernels.Cω(nothing, n1′, n2′, HelioseismicKernels.los_earth());
		@test C[:,1] ≈ C1
		@test C[:,2] ≈ C2
	end
	@testset "rotated biposh flipped" begin
		SHModes = LM(0:10, 0:10)
		jₒjₛ_allmodes = L2L1Triangle(1:4, 3, 1:4)
		Y12, Y21, Y1′2′, Y2′1′ = HelioseismicKernels.los_projected_spheroidal_biposh_flippoints(
			(n1, n1′), (n2, n2′), HelioseismicKernels.los_earth(), SHModes, jₒjₛ_allmodes);
		Y12_2, Y21_2 = HelioseismicKernels.los_projected_spheroidal_biposh_flippoints(
			n1, n2, HelioseismicKernels.los_earth(), SHModes, jₒjₛ_allmodes);
		Y1′2′_2, Y2′1′_2 = HelioseismicKernels.los_projected_spheroidal_biposh_flippoints(
			n1′, n2′, HelioseismicKernels.los_earth(), SHModes, jₒjₛ_allmodes);

		@test all(all(isapprox(Y1[i], Y2[i], atol=1e-14, rtol=1e-8) for i in eachindex(Y1)) for (Y1,Y2) in zip(Y1′2′, Y1′2′_2))
		@test all(all(isapprox(Y1[i], Y2[i], atol=1e-14, rtol=1e-8) for i in eachindex(Y1)) for (Y1,Y2) in zip(Y2′1′, Y2′1′_2))
		@test all(all(isapprox(Y1[i], Y2[i], atol=1e-14, rtol=1e-8) for i in eachindex(Y1)) for (Y1,Y2) in zip(Y12, Y12_2))
		@test all(all(isapprox(Y1[i], Y2[i], atol=1e-14, rtol=1e-8) for i in eachindex(Y1)) for (Y1,Y2) in zip(Y21, Y21_2))
	end
	@testset "kernel components" begin
		@testset "SoundSpeed" begin
			function uniformvalidation(::HelioseismicKernels.SoundSpeed, m::HelioseismicKernels.SeismicMeasurement, xobs1, xobs2,
				los::HelioseismicKernels.los_direction = HelioseismicKernels.los_radial(); SHModes = LM(0:0, 0:0))

				ℓ_range = 1:1

				hω = HelioseismicKernels.hω(nothing, m, xobs1, xobs2, los; ℓ_range = ℓ_range)

				Kδcₛₜ = HelioseismicKernels.kernel_δcₛₜ(nothing, m, xobs1, xobs2, los; ℓ_range = ℓ_range, SHModes = SHModes, hω=hω, save = false);

				Kδc₀₀ = HelioseismicKernels.kernel_δc₀₀(nothing, m, xobs1, xobs2, los; ℓ_range = ℓ_range, hω=hω, save = false);

				@test Kδcₛₜ[:,(0,0)]≈Kδc₀₀
			end

			@testset "TravelTimes" begin
				@testset "radial" begin
					uniformvalidation(HelioseismicKernels.SoundSpeed(), HelioseismicKernels.TravelTimes(), xobs1, xobs2, HelioseismicKernels.los_radial())
					uniformvalidation(HelioseismicKernels.SoundSpeed(), HelioseismicKernels.TravelTimes(), xobs1, xobs2, HelioseismicKernels.los_radial(),
						SHModes = LM(0:2, 0:2))
				end
				@testset "line-of-sight" begin
					uniformvalidation(HelioseismicKernels.SoundSpeed(), HelioseismicKernels.TravelTimes(), xobs1, xobs2, HelioseismicKernels.los_earth())
					uniformvalidation(HelioseismicKernels.SoundSpeed(), HelioseismicKernels.TravelTimes(), xobs1, xobs2, HelioseismicKernels.los_earth(),
						SHModes = LM(0:2, 0:2))
				end
			end
			@testset "Amplitudes" begin
				@testset "radial" begin
					uniformvalidation(HelioseismicKernels.SoundSpeed(), HelioseismicKernels.Amplitudes(), xobs1, xobs2, HelioseismicKernels.los_radial())
					uniformvalidation(HelioseismicKernels.SoundSpeed(), HelioseismicKernels.Amplitudes(), xobs1, xobs2, HelioseismicKernels.los_radial(),
						SHModes = LM(0:2, 0:2))
				end
				@testset "line-of-sight" begin
					uniformvalidation(HelioseismicKernels.SoundSpeed(), HelioseismicKernels.Amplitudes(), xobs1, xobs2, HelioseismicKernels.los_earth())
					uniformvalidation(HelioseismicKernels.SoundSpeed(), HelioseismicKernels.Amplitudes(), xobs1, xobs2, HelioseismicKernels.los_earth(),
						SHModes = LM(0:2, 0:2))
				end
			end
		end

		@testset "flows" begin
			function uniformvalidation(::HelioseismicKernels.Flow, m::HelioseismicKernels.SeismicMeasurement, xobs1, xobs2,
				los::HelioseismicKernels.los_direction = HelioseismicKernels.los_radial();
				SHModes = LM(1:1, 0:0))

				ℓ_range = 1:1

				hω = HelioseismicKernels.hω(nothing, m, xobs1, xobs2, los; ℓ_range = ℓ_range)

				Krθϕₛ₀ = HelioseismicKernels.kernel_uₛ₀_rθϕ(nothing, m, xobs1, xobs2, los; hω=hω, SHModes = SHModes,ℓ_range = ℓ_range, save = false)
				Krθϕₛ₀_2 = HelioseismicKernels.kernel_uₛ₀_rθϕ_2(nothing, m, xobs1, xobs2, los; hω=hω, SHModes = SHModes,ℓ_range = ℓ_range, save = false)

				Kℑu⁺₁₀ = HelioseismicKernels.kernel_ℑu⁺₁₀(nothing, m, xobs1, xobs2, los; ℓ_range = ℓ_range, hω=hω, save = false);
				@testset "Ks0 from Kst" begin
					@test @view(Krθϕₛ₀[:,3,(1,0)]) ≈ Kℑu⁺₁₀
				end
				@testset "Ks0 directly" begin
					@test @view(Krθϕₛ₀_2[:,3,(1,0)]) ≈ Kℑu⁺₁₀
				end
			end

			@testset "TravelTimes" begin
				@testset "radial" begin
					uniformvalidation(HelioseismicKernels.Flow(), HelioseismicKernels.TravelTimes(), xobs1, xobs2, HelioseismicKernels.los_radial())
					uniformvalidation(HelioseismicKernels.Flow(), HelioseismicKernels.TravelTimes(), xobs1, xobs2, HelioseismicKernels.los_radial(),
						SHModes = LM(1:2, 0:2))
				end
				@testset "line-of-sight" begin
					uniformvalidation(HelioseismicKernels.Flow(), HelioseismicKernels.TravelTimes(), xobs1, xobs2, HelioseismicKernels.los_earth())
					uniformvalidation(HelioseismicKernels.Flow(), HelioseismicKernels.TravelTimes(), xobs1, xobs2, HelioseismicKernels.los_earth(),
						SHModes = LM(1:2, 0:2))
				end
			end

			@testset "Amplitudes" begin
				@testset "radial" begin
					uniformvalidation(HelioseismicKernels.Flow(), HelioseismicKernels.Amplitudes(), xobs1, xobs2, HelioseismicKernels.los_radial())
					uniformvalidation(HelioseismicKernels.Flow(), HelioseismicKernels.Amplitudes(), xobs1, xobs2, HelioseismicKernels.los_radial(),
						SHModes = LM(1:2, 0:2))
				end
				@testset "line-of-sight" begin
					uniformvalidation(HelioseismicKernels.Flow(), HelioseismicKernels.Amplitudes(), xobs1, xobs2, HelioseismicKernels.los_earth())
					uniformvalidation(HelioseismicKernels.Flow(), HelioseismicKernels.Amplitudes(), xobs1, xobs2, HelioseismicKernels.los_earth(),
						SHModes = LM(1:2, 0:2))
				end
			end
		end
	end;

	@testset "Ks0 computed directly and from Kst" begin

		function test(m, los; SHModes = LM(1:1, 0:0), ℓ_range = 2:2)
			hω = HelioseismicKernels.hω(nothing, m, xobs1, xobs2, los; ℓ_range = ℓ_range)

			Krθϕₛ₀ = HelioseismicKernels.kernel_uₛ₀_rθϕ(nothing, m, xobs1, xobs2, los; hω=hω,
						SHModes = SHModes, ℓ_range = ℓ_range, save = false)
			Krθϕₛ₀_2 = HelioseismicKernels.kernel_uₛ₀_rθϕ_2(nothing, m, xobs1, xobs2, los; hω=hω,
						SHModes = SHModes, ℓ_range = ℓ_range, save = false)

			@test Krθϕₛ₀ ≈ Krθϕₛ₀_2
		end

		test(HelioseismicKernels.TravelTimes(), HelioseismicKernels.los_radial())
		test(HelioseismicKernels.TravelTimes(), HelioseismicKernels.los_radial(), SHModes = LM(0:4, 0:4))
		test(HelioseismicKernels.TravelTimes(), HelioseismicKernels.los_earth())
		test(HelioseismicKernels.TravelTimes(), HelioseismicKernels.los_earth(), SHModes = LM(0:4, 0:4))
	end;

	@testset "multiple points" begin
		@testset "flows" begin
			function testmultiplepoints(m, los)
				ℓ_range = 1:1
				hω = HelioseismicKernels.hω(nothing, m, xobs1, xobs2, los; ℓ_range = ℓ_range)
				K = HelioseismicKernels.kernel_ℑu⁺₁₀(nothing, m, n1, n2, los,ℓ_range = ℓ_range, hω = hω,
					ν_ind_range = axes(hω, 1));
				K2 = HelioseismicKernels.kernel_ℑu⁺₁₀(nothing, m, n1,[n2], los,ℓ_range = ℓ_range, hω = hω,
					ν_ind_range = axes(hω, 1));
				@test K ≈ K2
			end

			@testset "TravelTimes" begin
				@testset "radial" begin
					testmultiplepoints(HelioseismicKernels.TravelTimes(), HelioseismicKernels.los_radial())
				end
				@testset "line-of-sight" begin
					testmultiplepoints(HelioseismicKernels.TravelTimes(), HelioseismicKernels.los_earth())
				end
			end
			@testset "Amplitudes" begin
				@testset "radial" begin
					testmultiplepoints(HelioseismicKernels.Amplitudes(), HelioseismicKernels.los_radial())
				end
				@testset "line-of-sight" begin
					testmultiplepoints(HelioseismicKernels.Amplitudes(), HelioseismicKernels.los_earth())
				end
			end
		end
		@testset "SoundSpeed" begin
			function testmultiplepoints(m, los)
				ℓ_range = 1:1
				hω = HelioseismicKernels.hω(nothing, m, xobs1, xobs2, los; ℓ_range = ℓ_range)
				K = HelioseismicKernels.kernel_δc₀₀(nothing, m, n1, n2, los, ℓ_range = ℓ_range, hω = hω,
					ν_ind_range = axes(hω, 1));
				K2 = HelioseismicKernels.kernel_δc₀₀(nothing, m, n1,[n2], los, ℓ_range = ℓ_range, hω = hω,
					ν_ind_range = axes(hω, 1));
				@test K ≈ K2
			end

			@testset "TravelTimes" begin
				@testset "radial" begin
					testmultiplepoints(HelioseismicKernels.TravelTimes(), HelioseismicKernels.los_radial())
				end
				@testset "line-of-sight" begin
					testmultiplepoints(HelioseismicKernels.TravelTimes(), HelioseismicKernels.los_earth())
				end
			end
			@testset "Amplitudes" begin
				@testset "radial" begin
					testmultiplepoints(HelioseismicKernels.Amplitudes(), HelioseismicKernels.los_radial())
				end
				@testset "line-of-sight" begin
					testmultiplepoints(HelioseismicKernels.Amplitudes(), HelioseismicKernels.los_earth())
				end
			end
		end
	end;
end;
