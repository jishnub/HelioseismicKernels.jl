abstract type los_direction end

struct los_radial <: los_direction end
struct los_earth <: los_direction end

Base.broadcastable(l::los_direction) = Ref(l)

#######################################################################################################

# covariant helicity components of eₓ
function line_of_sight_covariant((θ,ϕ)::Tuple{Real,Real},
	los::los_earth = los_earth(), eulerangles::Nothing = nothing)

	m1 = +1/√2 * (cos(θ)cos(ϕ) + im*sin(ϕ))
	z  = sin(θ)cos(ϕ)
	p1 = -1/√2 * (cos(θ)cos(ϕ) - im*sin(ϕ))
	OffsetVector(SVector{3}(m1, z, p1), -1:1)
end

function rotatedpt((θ,ϕ), eulerangles::NTuple{3,Real})
	α, β, γ = eulerangles
	invrot = inv(RotZYZ(α, β, γ)) # active equivalent
	v = invrot * radiusvector(θ, ϕ)
	p = Point2D(v)
	p.θ, p.ϕ
end

function line_of_sight_covariant((θ,ϕ)::NTuple{2,Real},
	los::los_earth, eulerangles::NTuple{3,Real})

	θ′,ϕ′ = rotatedpt((θ,ϕ), eulerangles)

	p1 = -1/√2 * (cos(θ′)cos(ϕ′) - im*sin(ϕ′))
	m1 = +1/√2 * (cos(θ′)cos(ϕ′) + im*sin(ϕ′))
	z  = sin(θ′)cos(ϕ′)
	OffsetVector(SVector{3}(m1, z, p1),-1:1)
end

function line_of_sight_covariant(n::SphericalPoint,
	los::los_earth = los_earth(), eulerangles = nothing)

	line_of_sight_covariant((n.θ,n.ϕ), los, eulerangles)
end

function line_of_sight_covariant(n, ::los_radial, args...)
	OffsetArray(SVector{3,Float64}(0,1,0), -1:1)
end
