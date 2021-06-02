struct Point3D{T} <: FieldVector{3,T}
	r :: T
	θ :: T
	ϕ :: T
end
Point3D(x...) = Point3D(promote(x...)...)
struct Point2D{T} <: FieldVector{2,T}
	θ :: T
	ϕ :: T
end
Point3D(x::Tuple) = Point3D(x...)
Point2D(x::Tuple) = Point2D(x...)
Point2D(x::T...) where {T} = throw(ArgumentError("Point2D requires two angular coordinates"))
Point3D(x::T...) where {T} = throw(ArgumentError("Point2D requires three angular coordinates"))
Point2D(x...) = Point2D(promote(x...)...)

Point3D(r, p::Point2D) = Point3D(r, p.θ, p.ϕ)
Point3D(p::Point2D) = Point3D(1, p)
Point2D(p::Point3D) = Point2D(p.θ, p.ϕ)
Point2D(p::Point2D) = p
Point3D(p::Point3D) = p

radiusvector(θ, ϕ) = SVector{3}(sin(θ)cos(ϕ), sin(θ)sin(ϕ), cos(θ))
radiusvector(r, θ, ϕ) = SVector{3}(sin(θ)cos(ϕ), sin(θ)sin(ϕ), cos(θ)) .* r
radiusvector(p::Point3D) = radiusvector(p.r, p.θ, p.ϕ)
radiusvector(p::Point2D) = radiusvector(p.θ, p.ϕ)

Point3D(x::AbstractVector) = Point3D(SVector{3}(x))
Point2D(x::AbstractVector) = Point2D(SVector{3}(x))
function polarangles(v)
    x, y, z = v
    θ = acos(z)
    if abs(z) == 1
        ϕ = 0.0
    else
        ϕ = atan(x, y)
    end
    promote(θ, ϕ)
end
Point3D(v::SVector{3}) = Point3D(norm(v), polarangles(v)...)
Point2D(v::SVector{3}) = Point2D(polarangles(normalize(v))...)

const SphericalPoint = Union{Point2D,Point3D}

Base.broadcastable(p::SphericalPoint) = Ref(p)

for f in [:cosχ, :∂θ₁cosχ, :∂θ₂cosχ, :∂ϕ₁cosχ, :∂ϕ₂cosχ]
    @eval $f(n1, n2) = $f(Point2D(n1), Point2D(n2))
end
cosχ(n1::SphericalPoint, n2::SphericalPoint) = radiusvector(Point2D(n1)) ⋅ radiusvector(Point2D(n2))
∂θ₁cosχ(n1::SphericalPoint, n2::SphericalPoint) = epsilon(cosχ(Point2D(Dual(n1.θ,1), n1.ϕ), n2))
∂θ₂cosχ(n1::SphericalPoint, n2::SphericalPoint) = epsilon(cosχ(n1, Point2D(Dual(n2.θ,1), n2.ϕ)))
∂ϕ₁cosχ(n1::SphericalPoint, n2::SphericalPoint) = epsilon(cosχ(Point2D(n1.θ, Dual(n1.ϕ,1)), n2))
∂ϕ₂cosχ(n1::SphericalPoint, n2::SphericalPoint) = epsilon(cosχ(n1, Point2D(n2.θ, Dual(n2.ϕ, 1))))

shiftϕ(p::Point2D, Δϕ) = Point2D(p.θ, p.ϕ + Δϕ)
shiftϕ(p::Point3D, Δϕ) = Point3D(p.r, p.θ, p.ϕ + Δϕ)
