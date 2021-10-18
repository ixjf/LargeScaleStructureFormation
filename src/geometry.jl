module Geometry

export Vector3D, norm

# TODO: use LinearAlgebra. defines norm()
# FIXME: any way to make this mutable and stay isbits?
# I need a mutable struct!
# Also any way to make this parametrizable in a way that is still isbits?
struct Vector3D
    dx::Float32
    dy::Float32
    dz::Float32
end

function Base.copy(vec::Vector3D)::Vector3D
    Vector3D(vec.dx, vec.dy, vec.dz)
end

function Base.:+(lhs::Vector3D, rhs::Vector3D)::Vector3D
    Vector3D(lhs.dx + rhs.dx, lhs.dy + rhs.dy, lhs.dz + rhs.dz)
end

function Base.:-(lhs::Vector3D, rhs::Vector3D)::Vector3D
    Vector3D(lhs.dx - rhs.dx, lhs.dy - rhs.dy, lhs.dz - rhs.dz)
end

function Base.:*(lhs::Float32, rhs::Vector3D)::Vector3D
    Vector3D(lhs*rhs.dx, lhs*rhs.dy, lhs*rhs.dz)
end

function Base.:*(lhs::Vector3D, rhs::Float32)::Vector3D
    rhs*lhs
end

function Base.:/(lhs::Vector3D, rhs::Float32)::Vector3D
    Vector3D(lhs.dx/rhs, lhs.dy/rhs, lhs.dz/rhs)
end

function norm(v::Vector3D)::Float32
    √(v.dx^2 + v.dy^2 + v.dz^2)
end

end # module