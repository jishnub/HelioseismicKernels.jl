module Directions

using PointsOnASphere
using OffsetArrays
using Rotations
using CoordinateTransformations

export los_direction
export los_radial
export los_earth

export line_of_sight_covariant

abstract type los_direction end

struct los_radial <: los_direction end
struct los_earth <: los_direction end

#######################################################################################################

# covariant helicity components of eₓ
function line_of_sight_covariant((θ,ϕ)::Tuple{Real,Real},
	los::los_earth=los_earth(),eulerangles::Nothing=nothing)

	p1 = -1/√2 * (cos(θ)cos(ϕ)-im*sin(ϕ))
	m1 = +1/√2 * (cos(θ)cos(ϕ)+im*sin(ϕ))
	z  = sin(θ)cos(ϕ)
	OffsetVector([m1,z,p1],-1:1)
	# OffsetArray([0,1,0],-1:1) # useful for debugging
end

function rotatedpt((θ,ϕ),eulerangles::NTuple{3,Real})
	α,β,γ = eulerangles
	invrot = inv(RotZYZ(α,β,γ))
	p = Spherical(1,ϕ,π/2-θ)
	pxyz = CartesianFromSpherical()(p)
	p′xyz = invrot*pxyz
	p′ = SphericalFromCartesian()(p′xyz)
	π/2 - p′.ϕ, p′.θ 
end

function line_of_sight_covariant((θ,ϕ)::NTuple{2,Real},
	los::los_earth,eulerangles::NTuple{3,Real})

	# In this case we compute the rotated point
	θ′,ϕ′ = rotatedpt((θ,ϕ),eulerangles)

	p1 = -1/√2 * (cos(θ′)cos(ϕ′)-im*sin(ϕ′))
	m1 = +1/√2 * (cos(θ′)cos(ϕ′)+im*sin(ϕ′))
	z  = sin(θ′)cos(ϕ′)
	OffsetVector([m1,z,p1],-1:1)
end

function line_of_sight_covariant(n::SphericalPoint,
	los::los_earth=los_earth(),eulerangles=nothing)

	line_of_sight_covariant((n.θ,n.ϕ),los,eulerangles)
end

function line_of_sight_covariant(n,los::los_radial,eulerangles=nothing)
	OffsetArray([0,1,0],-1:1)
end

end # module