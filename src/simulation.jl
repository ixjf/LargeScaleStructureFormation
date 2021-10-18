module Simulation

#using StaticArrays
using CUDA
using DataStructures
using BenchmarkTools
using Distributed
using FFTW
using ..CDM.Snapshots:Snapshot
using ..CDM.Geometry

export run

const G_cgs = Float32(6.6743e-8) # CGS units

#CUDA.allowscalar(false)

# # TODO: is there a significant error caused by using Float32, seeing as it only has 7 digits of precision,
# # compared to 16 in a Float64?

struct BBox3D
    # TODO: mesh code needs to be updated since Vector3D is concrete Float64 type now
    #swf::Vector3D{Float32} # south-west front point
    swf::Vector3D
    side::Float32 # side length
end

# function contains(bbox::BBox3D, point::Vector3D{Float32})::Bool
#     (point.dx >= bbox.swf.dx && point.dx <= bbox.swf.dx + bbox.side) &&
#     (point.dy >= bbox.swf.dy && point.dy <= bbox.swf.dy + bbox.side) &&
#     (point.dz >= bbox.swf.dz && point.dz <= bbox.swf.dz + bbox.side)
# end

# function swf_quad(bbox::BBox3D)::BBox3D
#     hs = bbox.side/2
#     BBox3D(bbox.swf, hs)
# end

# function swb_quad(bbox::BBox3D)::BBox3D
#     hs = bbox.side/2
#     BBox3D(bbox.swf + Vector3D(Float32(0.0), hs, Float32(0.0)), hs)
# end

# function nwf_quad(bbox::BBox3D)::BBox3D
#     hs = bbox.side/2
#     BBox3D(bbox.swf + Vector3D(Float32(0.0), Float32(0.0), hs), hs)
# end

# function nwb_quad(bbox::BBox3D)::BBox3D
#     hs = bbox.side/2
#     BBox3D(bbox.swf + Vector3D(Float32(0.0), hs, hs), hs)
# end

# function sef_quad(bbox::BBox3D)::BBox3D
#     hs = bbox.side/2
#     BBox3D(bbox.swf + Vector3D(hs, Float32(0.0), Float32(0.0)), hs)
# end

# function seb_quad(bbox::BBox3D)::BBox3D
#     hs = bbox.side/2
#     BBox3D(bbox.swf + Vector3D(hs, hs, Float32(0.0)), hs)
# end

# function nef_quad(bbox::BBox3D)::BBox3D
#     hs = bbox.side/2
#     BBox3D(bbox.swf + Vector3D(hs, Float32(0.0), hs), hs)
# end

# function neb_quad(bbox::BBox3D)::BBox3D
#     hs = bbox.side/2
#     BBox3D(bbox.swf + Vector3D(hs, hs, hs), hs)
# end

# struct BHParticle
#     id::Union{Int64, Nothing} # if id != nothing, then this is a real particle. else, it's a center of mass
#     mass::Float32
#     coords::Vector3D{Float32}
# end

# mutable struct BHTreeNode
#     cm::Union{BHParticle, Nothing} # if it has children, it must have a center of mass. if not, it may be empty or have one particle attached
#     children::Union{SVector{8, BHTreeNode}, Nothing}
#     bbox::BBox3D

#     BHTreeNode(bbox::BBox3D) = new(nothing, nothing, bbox)
# end

# function is_external_node(bhnode::BHTreeNode)::Bool
#     isnothing(bhnode.children) && !isnothing(bhnode.cm)
# end

# function isempty(bhnode::BHTreeNode)::Bool
#     isnothing(bhnode.cm) && isnothing(bhnode.children)
# end

# function insert!(bhnode::BHTreeNode, particle::BHParticle)
#     if isnothing(bhnode.children)
#         if isnothing(bhnode.cm) # the node is empty
#             bhnode.cm = particle;
#         else
#             bhnode.children = SVector(BHTreeNode(nwf_quad(bhnode.bbox)),
#                                       BHTreeNode(nef_quad(bhnode.bbox)),
#                                       BHTreeNode(nwb_quad(bhnode.bbox)),
#                                       BHTreeNode(neb_quad(bhnode.bbox)),
#                                       BHTreeNode(swf_quad(bhnode.bbox)),
#                                       BHTreeNode(swb_quad(bhnode.bbox)),
#                                       BHTreeNode(sef_quad(bhnode.bbox)),
#                                       BHTreeNode(seb_quad(bhnode.bbox)))

#             insert!(bhnode, bhnode.cm)
#             insert!(bhnode, particle)
#         end
#     else
#         for bhchild in bhnode.children
#             if contains(bhchild.bbox, particle.coords)
#                 insert!(bhchild, particle)
#             end
#         end
#         # TODO: error if particle doesn't fit anywhere
#     end
# end

# # Calculates the centers of mass for the node and each of its children.
# # No need to calculate centers of mass and then propagate upwards EVERY time
# # we insert a new node
# function update_centers_of_mass!(bhnode::BHTreeNode)
#     if !isnothing(bhnode.children)
#         cm_coords = Vector3D(Float32(0.0), Float32(0.0), Float32(0.0))
#         total_mass = Float32(0.0)

#         for bhchild in bhnode.children
#             update_centers_of_mass!(bhchild)

#             # Some nodes may be empty
#             if !isnothing(bhchild.cm)
#                 cm_coords += bhchild.cm.mass*bhchild.cm.coords
#                 total_mass += bhchild.cm.mass
#             end
#         end

#         cm_coords /= total_mass

#         bhnode.cm = BHParticle(nothing, total_mass, cm_coords);
#     end
# end

# function draw_bhnode(bhnode::BHTreeNode)
#     # Draw bounding box
#     lower_left_f = bhnode.bbox.swf
#     lower_right_f = bhnode.bbox.swf + Vector3D(bhnode.bbox.side, Float32(0.0), Float32(0.0))
#     lower_left_b = bhnode.bbox.swf + Vector3D(Float32(0.0), bhnode.bbox.side, Float32(0.0))
#     lower_right_b = bhnode.bbox.swf + Vector3D(bhnode.bbox.side, bhnode.bbox.side, Float32(0.0))
#     upper_left_f = bhnode.bbox.swf + Vector3D(Float32(0.0), Float32(0.0), bhnode.bbox.side)
#     upper_right_f = bhnode.bbox.swf + Vector3D(bhnode.bbox.side, Float32(0.0), bhnode.bbox.side)
#     upper_left_b = bhnode.bbox.swf + Vector3D(Float32(0.0), bhnode.bbox.side, bhnode.bbox.side)
#     upper_right_b = bhnode.bbox.swf + Vector3D(bhnode.bbox.side, bhnode.bbox.side, bhnode.bbox.side)

#     plot!([lower_left_f.dx, lower_left_b.dx], [lower_left_f.dy, lower_left_b.dy], [lower_left_f.dz, lower_left_b.dz], seriestype=:line)
#     plot!([lower_right_f.dx, lower_right_b.dx], [lower_right_f.dy, lower_right_b.dy], [lower_right_f.dz, lower_right_b.dz], seriestype=:line)
#     plot!([upper_left_f.dx, upper_left_b.dx], [upper_left_f.dy, upper_left_b.dy], [upper_left_f.dz, upper_left_b.dz], seriestype=:line)
#     plot!([upper_right_f.dx, upper_right_b.dx], [upper_right_f.dy, upper_right_b.dy], [upper_right_f.dz, upper_right_b.dz], seriestype=:line)
#     plot!([lower_left_f.dx, lower_right_f.dx], [lower_left_f.dy, lower_right_f.dy], [lower_left_f.dz, lower_right_f.dz], seriestype=:line)
#     plot!([upper_left_f.dx, upper_right_f.dx], [upper_left_f.dy, upper_right_f.dy], [upper_left_f.dz, upper_right_f.dz], seriestype=:line)
#     plot!([lower_left_b.dx, lower_right_b.dx], [lower_left_b.dy, lower_right_b.dy], [lower_left_b.dz, lower_right_b.dz], seriestype=:line)
#     plot!([upper_left_b.dx, upper_right_b.dx], [upper_left_b.dy, upper_right_b.dy], [upper_left_b.dz, upper_right_b.dz], seriestype=:line)
#     plot!([lower_left_f.dx, upper_left_f.dx], [lower_left_f.dy, upper_left_f.dy], [lower_left_f.dz, upper_left_f.dz], seriestype=:line)
#     plot!([lower_right_f.dx, upper_right_f.dx], [lower_right_f.dy, upper_right_f.dy], [lower_right_f.dz, upper_right_f.dz], seriestype=:line)
#     plot!([lower_left_b.dx, upper_left_b.dx], [lower_left_b.dy, upper_left_b.dy], [lower_left_b.dz, upper_left_b.dz], seriestype=:line)
#     plot!([lower_right_b.dx, upper_right_b.dx], [lower_right_b.dy, upper_right_b.dy], [lower_right_b.dz, upper_right_b.dz], seriestype=:line)

#     # Draw children nodes
#     if !isnothing(bhnode.children)
#         for bhchild in bhnode.children
#             draw_bhnode(bhchild)
#         end
#     end

#     # Draw center of mass
#     if !isnothing(bhnode.cm)
#         scatter!([bhnode.cm.coords.dx], [bhnode.cm.coords.dy], [bhnode.cm.coords.dz], leg=nothing, markersize=0.5, markercolor=:red)
#     end
# end

# Note: We attempt a particle-based approach to particle-mesh spreading, as according to
# https://core.ac.uk/download/pdf/82127945.pdf
# this method is nowadays comparable in performance to more complicated methods

mutable struct Mesh
    size::Int64
    ρ::CuArray{Float32, 3} 
    ϕ::CuArray{Float32, 3}

    Mesh(Ng::Int64) = new(
        Ng, 
        CuArray{Float32, 3}(undef, Ng, Ng, Ng),
        CuArray{Float32, 3}(undef, Ng, Ng, Ng))
end

function compute_density_from_particle_list_kernel!(mesh_size::Int64, ρ::CuDeviceArray{Float32, 3}, p_mass::Float32, p_coords::CuDeviceArray{Vector3D, 1})
    index = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    for p_i=index:stride:length(p_coords) # Fun fact: if index > length(p_coords), simply nothing happens
        # Perform Cloud-In-Cell spreading
        # Find parent cell
        @inbounds i,j,k = Int64(floor(p_coords[p_i].dx)), Int64(floor(p_coords[p_i].dy)), Int64(floor(p_coords[p_i].dz)) # coords are in units of cell size, i.e. one cell size = 1 unit
        # thus if we have dx = 1, we have 0 + one cell size

        # Spread
        x_c,y_c,z_c = i,j,k

        @inbounds d_x = p_coords[p_i].dx - i
        @inbounds d_y = p_coords[p_i].dy - j
        @inbounds d_z = p_coords[p_i].dz - k

        t_x = 1 - d_x
        t_y = 1 - d_y
        t_z = 1 - d_z

        i_1 = i + 1
        j_1 = j + 1
        k_1 = k + 1

        # Offset indices (Julia is 1-index based)
        i += 1
        i_1 += 1
        j += 1
        j_1 += 1
        k += 1
        k_1 += 1

        i = enforce_periodic_mesh_coordinates(mesh_size, i)
        j = enforce_periodic_mesh_coordinates(mesh_size, j)
        k = enforce_periodic_mesh_coordinates(mesh_size, k)
        i_1 = enforce_periodic_mesh_coordinates(mesh_size, i_1)
        j_1 = enforce_periodic_mesh_coordinates(mesh_size, j_1)
        k_1 = enforce_periodic_mesh_coordinates(mesh_size, k_1)

        @inbounds @atomic ρ[i,j,k] += p_mass*t_x*t_y*t_z
        @inbounds @atomic ρ[i_1,j,k] += p_mass*d_x*t_y*t_z
        @inbounds @atomic ρ[i,j_1,k] += p_mass*t_x*d_y*t_z
        @inbounds @atomic ρ[i_1,j_1,k] += p_mass*d_x*d_y*t_z
        @inbounds @atomic ρ[i,j,k_1] += p_mass*t_x*t_y*d_z
        @inbounds @atomic ρ[i_1,j,k_1] += p_mass*d_x*t_y*d_z
        @inbounds @atomic ρ[i,j_1,k_1] += p_mass*t_x*d_y*d_z
        @inbounds @atomic ρ[i_1,j_1,k_1] += p_mass*d_x*d_y*d_z
    end
end

function compute_density_on_mesh!(mesh::Mesh, p_mass::Float32, p_coords::CuArray{Vector3D, 1})
    # Clear the density array, subtracting the mean "density"
    # remember that in the Poisson eq. I have the overdensity,
    # which after adimensionalization becomes tilde(ρ) - 1
    mesh.ρ = CUDA.fill(-1.0f0, mesh.size, mesh.size, mesh.size)
    
    numblocks = ceil(Int, length(p_coords)/128)
    CUDA.@sync begin
        @cuda threads=128 blocks=numblocks compute_density_from_particle_list_kernel!(
            mesh.size, 
            mesh.ρ, 
            p_mass, 
            p_coords)
    end
end

function green(l::Int64, m::Int64, n::Int64, box_size::Int64, Ωₘ₀::Float32, a::Float32)::Float64
    k_x = 2π*l/box_size
    k_y = 2π*m/box_size
    k_z = 2π*n/box_size
    -3Ωₘ₀/(8a*(CUDA.sin(k_x/2)^2 + CUDA.sin(k_y/2)^2 + CUDA.sin(k_z/2)^2))
end

function f(a::Float32, Ωₘ₀::Float32, Ωₖ₀::Float32, ΩΛ₀::Float32)::Float32
    CUDA.pow(1/((1/a)*(Ωₘ₀ + Ωₖ₀*a + ΩΛ₀*CUDA.pow(a, 3))), 0.5f0)
end

function g_x(ϕ::CuDeviceArray{Float32, 3, 1}, mesh_size::Int64, i::Int64, j::Int64, k::Int64)::Float32
    # i,j,k should be 1-based
    i_p1 = enforce_periodic_mesh_coordinates(mesh_size, i+1)
    i_m1 = enforce_periodic_mesh_coordinates(mesh_size, i-1)
    j = enforce_periodic_mesh_coordinates(mesh_size, j)
    k = enforce_periodic_mesh_coordinates(mesh_size, k)
    @inbounds -(ϕ[i_p1,j,k] - ϕ[i_m1,j,k])/2
end

function g_y(ϕ::CuDeviceArray{Float32, 3, 1}, mesh_size::Int64, i::Int64, j::Int64, k::Int64)::Float32
    # i,j,k should be 1-based
    i = enforce_periodic_mesh_coordinates(mesh_size, i)
    j_p1 = enforce_periodic_mesh_coordinates(mesh_size, j+1)
    j_m1 = enforce_periodic_mesh_coordinates(mesh_size, j-1)
    k = enforce_periodic_mesh_coordinates(mesh_size, k)
    @inbounds -(ϕ[i,j_p1,k] - ϕ[i,j_m1,k])/2
end

function g_z(ϕ::CuDeviceArray{Float32, 3, 1}, mesh_size::Int64, i::Int64, j::Int64, k::Int64)::Float32
    # i,j,k should be 1-based
    i = enforce_periodic_mesh_coordinates(mesh_size, i)
    j = enforce_periodic_mesh_coordinates(mesh_size, j)
    k_p1 = enforce_periodic_mesh_coordinates(mesh_size, k+1)
    k_m1 = enforce_periodic_mesh_coordinates(mesh_size, k-1)
    @inbounds -(ϕ[i,j,k_p1] - ϕ[i,j,k_m1])/2
end

function compute_fourier_space_gradient_kernel!(
        mesh_size::Int64, 
        fft_ϕ::CuDeviceArray{Complex{Float32}}, 
        fft_ρ::CuDeviceArray{Complex{Float32}}, 
        Ωₘ₀::Float32, 
        a::Float32)
    l_start = (blockIdx().x - 1)*blockDim().x + threadIdx().x - 1
    m_start = (blockIdx().y - 1)*blockDim().y + threadIdx().y - 1 
    n_start = (blockIdx().z - 1)*blockDim().z + threadIdx().z - 1
    l_stride = blockDim().x*gridDim().x
    m_stride = blockDim().y*gridDim().y
    n_stride = blockDim().z*gridDim().z

    # FIXME: is this kind of loop correct? does it iterate over each element EXACTLY once?
    # my quick tests showed that it does, but check again

    for l=l_start:l_stride:mesh_size - 1, 
        m=m_start:m_stride:mesh_size - 1, 
        n=n_start:n_stride:mesh_size - 1
        # TODO: can I make this branch disappear? does it even affect
        # performance?
        if l == 0 && m == 0 && n == 0
            continue
        end

        @inbounds fft_ϕ[l+1,m+1,n+1] = green(l, m, n, mesh_size, Ωₘ₀, a)*fft_ρ[l+1,m+1,n+1]
    end
end

function compute_gradient_on_mesh!(mesh::Mesh, Ωₘ₀::Float32, a::Float32)
    # FIXME: use rfft/irfft instead? plan fft?
    # Solve Poisson eq. using FFT
    fft_ρ = CUFFT.fft(mesh.ρ)

    fft_ϕ = CUDA.zeros(Complex{Float32}, mesh.size, mesh.size, mesh.size)

    fft_ϕ[1, 1, 1] = 0.0f0 # average value of ϕ in periodic box, singularity
    # called the "zero mode"

    begin
        numblocks = ceil(Int, mesh.size^3/128)
        CUDA.@sync begin
            @cuda threads=128 blocks=numblocks compute_fourier_space_gradient_kernel!(
                mesh.size,
                fft_ϕ,
                fft_ρ,
                Ωₘ₀,
                a)
        end
    end

    mesh.ϕ = CUFFT.real(CUFFT.ifft(fft_ϕ))
end

# i \in [1, mesh_size]
function enforce_periodic_mesh_coordinates(mesh_size::Int64, i::Int64)
    # FIXME: called in GPU. could this branching cause problems?
    if i <= 0
        i = mesh_size + i
    end

    (i - 1) % mesh_size + 1
end

function compute_acceleration_from_mesh(ϕ::CuDeviceArray{Float32, 3, 1}, mesh_size::Int64, p_coords::Vector3D)::Vector3D
    # CIC interpolation of potential to particle
    # TODO: this is the same code as above, fixme
    ## Find parent cell
    i,j,k = Int64(floor(p_coords.dx)), Int64(floor(p_coords.dy)), Int64(floor(p_coords.dz)) # coords are in units of cell size, i.e. one cell size = 1 unit
    # thus if we have dx = 1, we have 0 + one cell size

    ## Interpolate
    x_c,y_c,z_c = i,j,k

    d_x = p_coords.dx - i
    d_y = p_coords.dy - j
    d_z = p_coords.dz - k

    t_x = 1 - d_x
    t_y = 1 - d_y
    t_z = 1 - d_z

    # NOTE: i,j,k, i_1,j_1,k_1 do not take into account mesh size (they could be out of bounds)
    # we enforce periodic boundary conditions in g_x, g_y, g_z
    i_1 = i + 1
    j_1 = j + 1
    k_1 = k + 1

    # Offset indices (Julia is 1-index based)
    i += 1
    i_1 += 1
    j += 1
    j_1 += 1
    k += 1
    k_1 += 1

    ## 
    acc_x = g_x(ϕ, mesh_size, i, j, k)*t_x*t_y*t_z +
            g_x(ϕ, mesh_size, i_1, j, k)*d_x*t_y*t_z +
            g_x(ϕ, mesh_size, i, j_1, k)*t_x*d_y*t_z +
            g_x(ϕ, mesh_size, i_1, j_1, k)*d_x*d_y*t_z +
            g_x(ϕ, mesh_size, i, j, k_1)*t_x*t_y*d_z +
            g_x(ϕ, mesh_size, i_1, j, k_1)*d_x*t_y*d_z +
            g_x(ϕ, mesh_size, i, j_1, k_1)*t_x*d_y*d_z +
            g_x(ϕ, mesh_size, i_1, j_1, k_1)*d_x*d_y*d_z

    acc_y = g_y(ϕ, mesh_size, i, j, k)*t_x*t_y*t_z +
            g_y(ϕ, mesh_size, i_1, j, k)*d_x*t_y*t_z +
            g_y(ϕ, mesh_size, i, j_1, k)*t_x*d_y*t_z +
            g_y(ϕ, mesh_size, i_1, j_1, k)*d_x*d_y*t_z +
            g_y(ϕ, mesh_size, i, j, k_1)*t_x*t_y*d_z +
            g_y(ϕ, mesh_size, i_1, j, k_1)*d_x*t_y*d_z +
            g_y(ϕ, mesh_size, i, j_1, k_1)*t_x*d_y*d_z +
            g_y(ϕ, mesh_size, i_1, j_1, k_1)*d_x*d_y*d_z

    acc_z = g_z(ϕ, mesh_size, i, j, k)*t_x*t_y*t_z +
            g_z(ϕ, mesh_size, i_1, j, k)*d_x*t_y*t_z +
            g_z(ϕ, mesh_size, i, j_1, k)*t_x*d_y*t_z +
            g_z(ϕ, mesh_size, i_1, j_1, k)*d_x*d_y*t_z +
            g_z(ϕ, mesh_size, i, j, k_1)*t_x*t_y*d_z +
            g_z(ϕ, mesh_size, i_1, j, k_1)*d_x*t_y*d_z +
            g_z(ϕ, mesh_size, i, j_1, k_1)*t_x*d_y*d_z +
            g_z(ϕ, mesh_size, i_1, j_1, k_1)*d_x*d_y*d_z

    Vector3D(acc_x, acc_y, acc_z)
end

function update_momentum_first_halfstep_euler_kernel!(
        ϕ::CuDeviceArray{Float32, 3, 1}, 
        mesh_size::Int64, 
        a::Float32, 
        Ωₘ₀::Float32, 
        Ωₖ₀::Float32, 
        ΩΛ₀::Float32, 
        Δa::Float32,
        x_iu::CuDeviceArray{Vector3D, 1}, 
        p_iu::CuDeviceArray{Vector3D, 1})
    index = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    for p_i=index:stride:length(x_iu)
        @inbounds accel = compute_acceleration_from_mesh(ϕ, mesh_size, x_iu[p_i])
        @inbounds p_iu[p_i] += f(a, Ωₘ₀, Ωₖ₀, ΩΛ₀)*accel*(Δa/2)
    end
end

function update_particles_kernel!(
        ϕ::CuDeviceArray{Float32, 3, 1}, 
        mesh_size::Int64, 
        a::Float32, 
        Ωₘ₀::Float32, 
        Ωₖ₀::Float32, 
        ΩΛ₀::Float32, 
        Δa::Float32, 
        bbox::BBox3D, 
        x_iu::CuDeviceArray{Vector3D, 1}, 
        p_iu::CuDeviceArray{Vector3D, 1})
    index = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    for p_i=index:stride:length(x_iu)
        @inbounds accel = compute_acceleration_from_mesh(ϕ, mesh_size, x_iu[p_i])
        #println(accel)

        # req_timestep = 0.3*√(1*a^3/norm(accel))

        # # ???
        # if req_timestep - Δa <= -Δa 
        #     timestep_100 += 1
        # elseif req_timestep - Δa <= -0.7Δa
        #     timestep_70 += 1
        # elseif req_timestep - Δa <= -0.5Δa
        #     timestep_50 += 1
        # elseif req_timestep - Δa <= -0.3Δa
        #     timestep_30 += 1
        # elseif req_timestep - Δa <= -0.1Δa
        #     timestep_10 += 1
        # end

        # min_timestep = min(min_timestep, req_timestep)

        # use kinematic time step criterion to check how many particles needed a smaller timestep
        # softening length = cell size
        # also grid size: can we increase grid size as much as we want, or is there a point in which
        # we get garbage results (from being too high) - because there is a massive difference in GADGET's
        # output when grid size = 384 and grid size = 16
        # why do we only start to see structure around z = 5?
        # experiment with GADGET with fixed timestep. see the effect and compare to my results.
        # TODO: optimize, optimize, optimize. see problems reported in report
        # TODO: first things first, review all derivations and theory before starting working on this again
        # e.g. why do we need to use the same interpolation scheme for both acceleration & density?
        # I mean, i know why. but why is it that that happens? why do self-forces arise? why is momentum
        # not conserved otherwise? I guess the basic idea is that, if the scheme is different, then you're
        # assuming particles have a certain contribution to the field in one step, but a different contribution
        # in another step, thus getting numerical errors
        # TODO: 1) test, debug, implement other TODOs
        # TODO: 2) merge with tree method, or see alternative pm methods
        #println("previous moment: $(p[p_i])")
        @inbounds p_iu[p_i] += f(a, Ωₘ₀, Ωₖ₀, ΩΛ₀)*accel*Δa # time n + 1/2
        #println("new moment: $(p[p_i])")

        a_n_plus_1_2 = a + 0.5f0*Δa
        @inbounds x_iu[p_i] += (1/a_n_plus_1_2^2)*f(a_n_plus_1_2, Ωₘ₀, Ωₖ₀, ΩΛ₀)*p_iu[p_i]*Δa # time n + 1
        #println("dx for $p_i: ", (1/a_n_plus_1_2^2)*f(a_n_plus_1_2, Ωₘ₀, Ωₖ₀, ΩΛ₀)*p[p_i]*Δa)

        #println("before: $(x[p_i])")
        @inbounds x_iu[p_i] = box_wrap(x_iu[p_i], bbox)
        #println("after: $(x[p_i])")
    end
end

# FIXME: hubble_param needs to be 64bit?
# TODO: these parameters should come from the initial conditions, no?
function run(ic::Snapshot, 
             output_times::Array{Float32, 1}, 
             output_fn,
             Ɛ::Float32, 
             η::Float32, 
             θ::Float32, 
             max_time_step::Float32,
             mesh_size::Int64,
             H₀::Float64, # CGS
             Ωₘ₀::Float32,
             Ωₖ₀::Float32,
             ΩΛ₀::Float32,
             Δa::Float32)
    # Define base units
    r₀ = ic.box_size*ic.unit_length_cgs/mesh_size # CGS
    t₀ = 1/H₀ # = seconds
    ρ₀ = 3H₀^2/(8π*G_cgs)*Ωₘ₀
    #ϕ₀ = r₀^2/t₀^2
    v₀ = r₀/t₀
    m₀ = ρ₀*r₀^3

    # Initialize state
    a = Float32(ic.time)
    p_mass_iu = Float32(ic.mass*ic.unit_mass_cgs/m₀) # mass of particles is total_mass = (Omega0*3*H0^2/(8*pi*G)*L^3)=Omega0*crit_density
    # m1 = total_mass/N_particles, so it doesn't depend on any parameters that I might change for my simulation
    # (like grid size).
    x_iu = CuArray(copy(ic.coords).*Float32(ic.unit_length_cgs/r₀))
    p_iu = CuArray(a^(1.5f0).*copy(ic.velocities).*Float32(ic.unit_vel_cgs/v₀))
    box_size_iu = Float32(ic.box_size*ic.unit_length_cgs/r₀) # = mesh_size in code units
    bbox = BBox3D(Vector3D(0.0, 0.0, 0.0), box_size_iu) # PBC box

    # NOTE: Two reasons why increasing the grid creates much more visible structure
    # 1) a finer grid means higher resolution
    # 2) a finer grid means greater mass per particle (wait what? lower mass!), thus greater contribution to the density field
    # we see that in GADGET as well. (this sounds nonsensical)
    # so the question is, how large should a grid be to get the most accurate
    # representation of the universe?

    mesh = Mesh(mesh_size) # initializes arrays for density and potential
    # reused between time steps to avoid unnecessary reallocations

    # Take the first step for momentum with Euler method
    # since the integrator is not self-starting
    # we need the momentum at initial_a + 1/2*Δa
    compute_density_on_mesh!(mesh, p_mass_iu, x_iu)
    compute_gradient_on_mesh!(mesh, Ωₘ₀, a)

    # FIXME: what's the point of a Mesh struct if I have to pass components of it separately to GPU kernels?

    begin
        numblocks = ceil(Int, length(x_iu)/128)
        CUDA.@sync begin
            @cuda threads=128 blocks=128 update_momentum_first_halfstep_euler_kernel!(mesh.ϕ, mesh_size, a, Ωₘ₀, Ωₖ₀, ΩΛ₀, Δa, x_iu, p_iu)
        end
    end

    # TODO: how to take global adaptive timestep instead of
    # fixed global timestep?

    # Set up output times stack
    output_times_s = Stack{Float32}()

    for output_time in sort(output_times, rev=true)
        push!(output_times_s, output_time) # last in, first out
    end

    # Setup info logs
    # tslog = open("timesteps.txt", "w")

    while a + Δa < ic.time_end # we don't want to do one step above a = 1.0
        # Check if need to output first (we may want to output at initial time, thus we need to output _now_)
        if !isempty(output_times_s)
            output_time = first(output_times_s)

            if a - 2Δa <= output_time <= a + 2Δa
                # there is an interval between a = 1.0 - Δa and a = 1.0
                # that may be overlooked because we do not take any further step if the timestep would be so
                # large it would go beyond a = 1.0
                # thus we use 2*delta_a as the criterion

                snap = copy(ic)
                snap.box_size = box_size_iu
                snap.time = a
                snap.mass = p_mass_iu
                snap.unit_length_cgs = r₀
                snap.unit_mass_cgs = m₀
                snap.unit_vel_cgs = v₀
                snap.coords = Array(x_iu)#.*r₀
                snap.velocities = Array(p_iu./a^(1.5f0))#.*v₀
                # all other values stayed the same. TODO right?
                
                pop!(output_times_s)
                
                # FIXME: crash here @spawn for some reason (HDF5 dataset can't get value)
                #Threads.@spawn 
                output_fn(snap)
            end
        end

        a += Δa

        #println(tslog, "Computing timestep a = $a")
        println("Computing timestep a = $a")

        # Perform CIC density assignment
        compute_density_on_mesh!(mesh, p_mass_iu, x_iu)

        # # Solve Poisson equation
        compute_gradient_on_mesh!(mesh, Ωₘ₀, a)

        # # TEST: Count number of particles with required timestep 10% smaller
        # # than the timestep used
        # # timestep_10 = 0
        # # timestep_30 = 0
        # # timestep_50 = 0
        # # timestep_70 = 0
        # # timestep_100 = 0
        # # min_timestep = Inf64

        # # Advance positions and velocities
        begin
            numblocks = ceil(Int, length(x_iu)/128)
            CUDA.@sync begin
                @cuda threads=128 blocks=numblocks update_particles_kernel!(
                    mesh.ϕ,
                    mesh.size,
                    a,
                    Ωₘ₀, 
                    Ωₖ₀, 
                    ΩΛ₀, 
                    Δa, 
                    bbox, 
                    x_iu, 
                    p_iu
                )
            end
        end

        # println(tslog, "------------------------------------------")
        # println(tslog, "Min required timestep (v/g) = $min_timestep")
        # println(tslog, "N particles required timestep >= 100% smaller than Δa: $timestep_100")
        # println(tslog, "N particles required timestep <= 70% Δa: $timestep_70")
        # println(tslog, "N particles required timestep <= 50% Δa: $timestep_50")
        # println(tslog, "N particles required timestep <= 30% Δa: $timestep_30")
        # println(tslog, "N particles required timestep <= 10% Δa: $timestep_10")
    end

    # close(tslog)

    # scatter([], [], [])

    # for coords in x_iu
    #     scatter!([coords.dx], [coords.dy], [coords.dz], leg=nothing, markersize=0.25)
    # end

    # xlims!((0.0, 16.0))
    # ylims!((0.0, 16.0))
    # zlims!((0.0, 16.0))

    # gui()



    # #finished = false

    # # TODO: there's information that doesn't need to be copied every time
    # state = copy(ic)

    # G_int_units = Float32(G_cgs*ic.unit_mass_cgs/(ic.unit_length_cgs*ic.unit_vel_cgs^2))

    # a = Array{Vector3D{Float32}, 1}(undef, length(state.particle_ids))
    # #ts= Array{Float32, 1}(undef, length(state.particle_ids))

    # dt = max_time_step

    # # Initial force computation
    # bhroot = compute_bhtree(state)

    # for (i, i_coords) in enumerate(state.coords)
    #     a[i] = compute_accel(i, i_coords, bhroot, G_int_units, Ɛ, θ)
    #     dt = min(max_time_step, η*√(Ɛ/norm(a[i])))
    # end

    # # FIXME: how to NOT duplicate computations of dt here and at end of while loop? :)
    # # Why is this much faster than per-particle time steps? weren't we doing the same
    # # amount of timesteps? probably because I changed the settings! still should take
    # # ~55min for 32^3 particles :)

    # #bhroot = nothing
    
    # #while !finished # while some particle has time < time_end, update those particles
    # while state.time < state.time_end
    #     state.time = min(state.time + dt, state.time_end)

    #     #println("dt: $(dt), time: $(state.time)")
        
    #     # TODO: take step of size max_time_step. do intermediate calculations of forces in the interval [t, t + max_time_step]
    #     # ...
        
    #     #finished = true
        
    #     # Particles have different timesteps because they do not move at the same rate. We assume
    #     # that the current position of all particles is the position at their respective last time because of this.
    #     # FIXME: but is this an adequate choice? NOPE!
        
    #     # Kick-drift-kick leapfrog integration (this is time-symmetric. certain quantities
    #     # like energy should be conserved to a very high degree - one should see
    #     # periodic total energy with ~constant amplitude for an orbital system)
    #     # First kick and drift all particles halfway through dt
    #     # Also KDK requires only 1 computation of force per timestep. DKD requires two.
    #     for i=1:length(state.particle_ids)
    #         #dt = η*√(Ɛ/norm(a[i])) # is the most common, not expensive since we already have to calculate
    #         # the acceleration. there are better criteria, but they don't really perform much better for large scale
    #         # structure formation with such low resolution (https://academic.oup.com/mnras/article/376/1/273/973561)
            
    #         #state.timesteps[i] = min(state.timesteps[i] + dt, ic.time_end)
    #         state.timesteps[i] = state.time
            
    #         #if state.timesteps[i] < ic.time_end
    #         #    finished = false
    #         #elseif state.timesteps[i] == ic.time_end
    #         #    continue
    #         #end
            
    #         state.velocities[i] += a[i]*dt/2 # actually this is v at dt/2, but we need this value for the next kick
    #         # and we don't need the previous velocity anyway
    #         state.coords[i] += state.velocities[i]*dt

    #         # Periodic boundary conditions - wrap particles in box (if they leave the box,
    #         # they come out the other side)
    #         box_wrap!(state.coords[i], bhroot.bbox)
            
    #         #ts[i] = dt
    #     end
        
    #     # Recompute tree since we moved the particles.
    #     bhroot = compute_bhtree(state)
        
    #     # Then kick again all the way through dt
    #     for i=1:length(state.particle_ids)
    #         # FIXME: there was a bug in the old adaptive per-particle timestepping.
    #         # it didn't compute velocities when == ic.time_end, so last time was wrong.
    #         #if state.timesteps[i] == ic.time_end
    #         #    continue
    #         #end
            
    #         a[i] = compute_accel(i, state.coords[i], bhroot, G_int_units, Ɛ, θ)
    #         state.velocities[i] += a[i]*dt/2#*ts[i]/2
            
    #         dt = min(max_time_step, η*√(Ɛ/norm(a[i])))
    #     end
    # end
    
    # scatter([], [], [])
    
    # for (i, i_coords) in enumerate(state.coords)
    #     scatter!([i_coords.dx], [i_coords.dy], [i_coords.dz], leg=nothing, markersize=1.0, markercolor=:blue, figsize=(1000, 1000))
    # end

    # draw_bhnode(bhroot)    

    # gui()
end

function box_wrap(coords::Vector3D, bbox::BBox3D)
    new_dx, new_dy, new_dz = coords.dx, coords.dy, coords.dz

    if coords.dx > bbox.swf.dx + bbox.side
        new_dx = bbox.swf.dx;
    end

    if coords.dx < 0
        new_dx = bbox.swf.dx + bbox.side;
    end

    if coords.dy > bbox.swf.dy + bbox.side
        new_dy = bbox.swf.dy;
    end

    if coords.dy < 0
        new_dy = bbox.swf.dy + bbox.side;
    end

    if coords.dz > bbox.swf.dz + bbox.side
        new_dz = bbox.swf.dz;
    end

    if coords.dz < 0
        new_dz = bbox.swf.dz + bbox.side;
    end

    Vector3D(new_dx, new_dy, new_dz)
end

# function compute_bhtree(state::Snapshot)
#     # FIXME: should I recalculate the box size every time? does this have any benefit?
#     # I think maybe not? if we have fully periodic boundary conditions,
#     # then the box doesn't grow larger than state.box_size
#     # And a larger root around all the particles is not much of a problem either
#     # (computing the bounding box every time would probably be more computationally
#     # expensive)
    
#     # max_x, max_y, max_z = Float32(0.0), Float32(0.0), Float32(0.0)
#     # min_x, min_y, min_z = Float32(0.0), Float32(0.0), Float32(0.0)
#     # for coords in state.coords
#     #     max_x = max(max_x, coords.dx)
#     #     max_y = max(max_y, coords.dy)
#     #     max_z = max(max_z, coords.dz)
#     #     min_x = min(min_x, coords.dx)
#     #     min_y = min(min_y, coords.dy)
#     #     min_z = min(min_z, coords.dz)
#     # end

#     # max_side = max(abs(min_x) + abs(max_x), abs(min_y) + abs(max_y), abs(min_z) + abs(max_z))

#     bhroot = BHTreeNode(
#         BBox3D(
#             Vector3D(Float32(0.0), Float32(0.0), Float32(0.0)),
#             state.box_size
#             #Vector3D(min_x, min_y, min_z), 
#             #max_side
#         )
#     )

#     for (i, i_coords) in enumerate(state.coords)
#         insert!(bhroot, BHParticle(i, state.mass, i_coords))
#     end

#     update_centers_of_mass!(bhroot)

#     bhroot
# end

# function compute_accel(i::Int64, 
#                        i_coords::Vector3D{Float32}, 
#                        bhnode::BHTreeNode, 
#                        G_int_units::Float32, 
#                        Ɛ::Float32, 
#                        θ::Float32)
#     if isempty(bhnode)
#         return Vector3D(Float32(0.0), Float32(0.0), Float32(0.0))
#     end

#     r_ji = i_coords - bhnode.cm.coords

#     if is_external_node(bhnode)
#         if bhnode.cm.id != i
#             # compute force on single particle
#             return -G_int_units*bhnode.cm.mass*r_ji/(norm(r_ji)^2 + Ɛ^2)^Float32(3/2)
#         else
#             return Vector3D(Float32(0.0), Float32(0.0), Float32(0.0))
#         end
#     else
#         # TODO: Check again if you got the ratio right. is it this node's side length?
#         # or something else?
#         ratio = bhnode.bbox.side/norm(r_ji)

#         if ratio < θ
#             # compute force on group of particles
#             return -G_int_units*bhnode.cm.mass*r_ji/(norm(r_ji)^2 + Ɛ^2)^Float32(3/2)
#         else
#             # sum accelerations due to each child
#             a_i = Vector3D(Float32(0.0), Float32(0.0), Float32(0.0))
#             for bhchild in bhnode.children # internal nodes ALWAYS have children
#                 a_i += compute_accel(i, i_coords, bhchild, G_int_units, Ɛ, θ)
#             end
#             return a_i
#         end
#     end
# end

end # module