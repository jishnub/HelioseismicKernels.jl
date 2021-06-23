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

function polcoords(v)
    # This is not unique at the poles, as (θ=0, ϕ) map to the north pole for all ϕ
	# We choose to retutn ϕ = 0 at the poles
	vunit = normalize(v)
	if vunit[3] == 1
		# north pole
		return SVector{2}(promote(oftype(float(vunit[3]), 0), 0))
	elseif vunit[3] == -1
		# south pole
		return SVector{2}(promote(oftype(float(vunit[3]), pi), 0))
	end
	θ = acos(vunit[3])
	ϕ = mod2pi(atan(vunit[2], vunit[1]))
	SVector{2}(promote(θ, ϕ))
end
rotn1n1′(θ1, ϕ1, θ1′, ϕ1′) = RotZYZ(ϕ1′, θ1′ - θ1, -ϕ1)
isapproxdefault(x, y) = isapprox(x, y, atol = 1e-14, rtol = 1e-8)
function rotn1′n2n2′(r1′, r2′′, r2′)
	r1′x, r1′y, r1′z = r1′
	a = r2′′ - r1′
	b = r2′ - r1′
	# Avoid errors due to machine precision
	if (isapproxdefault(cross(r1′, a), zero(SVector{3})) || isapproxdefault(cross(r1′, b), zero(SVector{3})))
		return RotZYZ(one(SMatrix{3,3,eltype(r1′),9}))
	end
	sinω = b ⋅ cross(r1′, a) / norm(cross(r1′, a))^2
	cosω = b ⋅ (a - (r1′⋅a)r1′) / norm(a - (r1′⋅a)r1′)^2
	r1′C = SMatrix{3,3}(0, r1′z, -r1′y, -r1′z, 0, r1′x, r1′y, -r1′x, 0)
	r1′r1′ = r1′ * r1′'
	RotZYZ(cosω*I + sinω * r1′C + (1 - cosω)*r1′r1′)
end

function _rotation_points(::Cartesian, n1, n2, n1′, n2′)
	(θ1, ϕ1), (θ1′, ϕ1′) = map(Point2D, (n1, n1′))
	r1, r2, r1′, r2′ = map(radiusvector∘Point2D	, (n1, n2, n1′, n2′))
	if !isapproxdefault(r1 ⋅ r2, r1′ ⋅ r2′)
		throw(ArgumentError("points are not related by a rotation"))
	end
	R11′ = rotn1n1′(θ1, ϕ1, θ1′, ϕ1′)
	r2′′ = R11′ * r2
	R2′′2′ = rotn1′n2n2′(r1′, r2′′, r2′)
	R = RotZYZ(R2′′2′ * R11′)
	R * r1 ≈ r1′ || error("rotation does not map r1 to r1′")
	R * r2 ≈ r2′ || error("rotation does not map r2 to r2′")
	return R
end

function _rotation_points(Basis, n1, n2, n1′, n2′)
	R = _rotation_points(Cartesian(), Point2D(n1), Point2D(n2), Point2D(n1′), Point2D(n2′))
	Un1, Un1′, Un2, Un2′ = map((n1, n1′, n2, n2′)) do n
		VectorSphericalHarmonics.basisconversionmatrix(Cartesian(), Basis, Point2D(n)...)
	end
	M′1 = Un1′ * R * Un1'
	M′2 = Un2′ * R * Un2'
	LinearAlgebra.kron(M′1, M′2)
end
rotation_points(Basis, n1, n2, n1′, n2′) = _rotation_points(Basis, n1, n2, n1′, n2′)
rotation_points(Basis::HelicityCovariant, n1, n2, n1′, n2′) = Diagonal(_rotation_points(Basis, n1, n2, n1′, n2′))
