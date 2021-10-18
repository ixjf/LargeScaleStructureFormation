module Snapshots

using Base.Filesystem
using HDF5
using ..CDM.Geometry:Vector3D

# TODO: what does export actually do?
export read_snap, write_snap, Snapshot

# TODO: Some unit tests would be useful here

# NVIDIA gaming GPUs only perform well with 16 & 32bit floats. We force GADGET to output
# in 32bit as well (or try to.)

mutable struct Snapshot
    #redshift::Float32 # z
    mass::Float64
    box_size::Float64
    time_begin::Float64 # internal units [0.0; 1.0] where t = 1/(1 + z)
    time::Float64
    time_end::Float64
    #
    unit_length_cgs::Float64
    unit_mass_cgs::Float64
    unit_vel_cgs::Float64
    #
    # particle_ids only actually useful if we're comparing two different snapshots (i.e. at different times)
    #particle_ids::Array{UInt32, 1} # assumes IDs were stored as 32bit, FIXME
    # index at particle_ids => particle ID
    # same index at coords, timesteps, velocities corresponds to same particle ID
    coords::Array{Vector3D, 1} # NOTE: these are comoving coordinates x
    timesteps::Array{Float32, 1} # assumes timesteps were stored as 32bit, FIXME
    velocities::Array{Vector3D, 1} # assumes velocities were stored as 32bit, FIXME
    # NOTE: velocities are in GADGET units u = v/sqrt(a), where v is peculiar velocity
    #accel::Dict{UInt32, Vector3D{Float32}} # same idea as above, FIXME
end

function Base.copy(s::Snapshot)::Snapshot
    Snapshot(#s.redshift,
             s.mass,
             s.box_size,
             s.time_begin,
             s.time,
             s.time_end,
             s.unit_length_cgs,
             s.unit_mass_cgs,
             s.unit_vel_cgs,
             #copy(s.particle_ids),
             copy(s.coords),
             copy(s.timesteps),
             copy(s.velocities)#,
             #copy(s.accel)
             )
end

# Reads a GADGET4 HDF5 snapshot (either IC or curr state at some z) with DM particles 
# Each snapshot file contains the info about particles at some z (and only one z)

function read_coordinates!(f::HDF5File, coords_out::Array{Vector3D, 1})
    for c in eachrow(read(f["/PartType1/Coordinates"])') # why did I need to transpose?
        # why does the result of read treat each row as a column instead?
        push!(coords_out, Vector3D(c[1], c[2], c[3]))
    end
end

function read_timesteps!(f::HDF5File, ts_out::Array{Float32, 1})
    for t in read(f["/PartType1/TimeStep"])
        push!(ts_out, t)
    end
end

function read_velocities!(f::HDF5File, v_out::Array{Vector3D, 1})
    for v in eachrow(read(f["/PartType1/Velocities"])')
        # FIXME: same problem with transpose as in read_coordinates!
        push!(v_out, Vector3D(v[1], v[2], v[3]))
    end
end

# function read_accelerations!(f::HDF5File, 
#                              particle_ids::Array{UInt32, 1}, 
#                              a_out::Dict{UInt32, Vector3D{Float32}}) # FIXME: 32bit, same as read_coordinates!
#     for (i,a) in enumerate(eachrow(read(f["/PartType1/Acceleration"])'))
#         # FIXME: same problem with transpose as in read_coordinates!
#         push!(a_out, particle_ids[i] => Vector3D(a[1], a[2], a[3]))
#     end
# end

function read_snap(snapfile::T)::Snapshot where {T <: AbstractString}
    # FIXME: will throw HDF5.jl-specific error if failed to open file
    #redshift = 0.0
    box_size = 0.0
    mass = 0.0
    time_begin = 0.0
    time = 0.0
    time_end = 0.0
    unit_length_cgs = 0.0
    unit_mass_cgs = 0.0
    unit_vel_cgs = 0.0
    #particle_ids = Array{UInt32, 1}()
    coords = Array{Vector3D, 1}()
    timesteps = Array{Float32, 1}()
    velocities = Array{Vector3D, 1}()
    #accel = Dict{UInt32, Vector3D{Float32}}()

    files_to_read = []
    
    h5open(snapfile, "r") do f
        # If comoving integration isn't enabled, fail
        comoving_int_on = convert(Bool, read(attributes(f["/Parameters"])["ComovingIntegrationOn"]))

        if !comoving_int_on
            error("snapshot must have been calculated with comoving integration on")
        end

        # Snapshot may be split into multiple files. Figure out how many and where they are located        
        begin
            num_files_per_snapshot = read(attributes(f["/Parameters"])["NumFilesPerSnapshot"])

            filename, ext = splitext(basename(snapfile))
            
            # Generate full paths to snapshot files
            dir = dirname(snapfile)
            for i=0:num_files_per_snapshot - 1 # .0, .1, .2, up to .(num_files_per_snapshot - 1)
                rem_filepath = joinpath(dir, filename[1:end - 2] * '.' * string(i) * ext)                
                push!(files_to_read, rem_filepath)
            end
        end

        # Read other properties
        #redshift = convert(Float32, read(attributes(f["/Header"])["Redshift"]))
        #box_size = convert(Float32, read(attributes(f["/Header"])["BoxSize"]))
        box_size = read(attributes(f["/Header"])["BoxSize"])
        #scaling_factor = read(attributes(f["/Header"])["Time"])
        #mass = convert(Float32, read(attributes(f["/Header"])["MassTable"])[2])
        mass = read(attributes(f["/Header"])["MassTable"])[2]

        # time_begin = convert(Float32, read(attributes(f["/Parameters"])["TimeBegin"]))
        # time = convert(Float32, read(attributes(f["/Header"])["Time"]))
        # time_end = convert(Float32, read(attributes(f["/Parameters"])["TimeMax"]))
        time_begin = read(attributes(f["/Parameters"])["TimeBegin"])
        time = read(attributes(f["/Header"])["Time"])
        time_end = read(attributes(f["/Parameters"])["TimeMax"])

        unit_length_cgs = read(attributes(f["/Parameters"])["UnitLength_in_cm"])
        unit_mass_cgs = read(attributes(f["/Parameters"])["UnitMass_in_g"])
        unit_vel_cgs = read(attributes(f["/Parameters"])["UnitVelocity_in_cm_per_s"])
    end

    for filepath in files_to_read
        h5open(filepath, "r") do f
            #append!(particle_ids, read(f["/PartType1/ParticleIDs"]))
            read_coordinates!(f, coords)
            read_timesteps!(f, timesteps)
            read_velocities!(f, velocities)
            #read_accelerations!(f, particle_ids, accel)
       end
    end

    # TODO: GADGET units -> CGS -> internal units and back might actually
    # cause overflows down the road
    # Change units to CGS
    #mass *= unit_mass_cgs
    #box_size *= unit_length_cgs
    #coords .*= unit_length_cgs
    #timesteps .*= unit_vel_cgs/unit_length_cgs
    #velocities .*= unit_vel_cgs

    Snapshot(#redshift, 
             mass, 
             box_size, 
             time_begin, 
             time,
             time_end, 
             unit_length_cgs, 
             unit_mass_cgs, 
             unit_vel_cgs, 
            # particle_ids,
             coords, 
             timesteps, 
             velocities#, 
             #accel
             )
end

function write_snap(snap::Snapshot, filepath::T) where {T <: AbstractString}
    h5open(filepath, "w") do f
        header = create_group(f, "Header")
        attributes(header)["Time"] = snap.time
        attributes(header)["Redshift"] = 1/snap.time - 1
        attributes(header)["NumPart_Total"] = [0, length(snap.coords)]
        attributes(header)["NumPart_ThisFile"] = [0, length(snap.coords)]
        attributes(header)["NumFilesPerSnapshot"] = 1
        attributes(header)["MassTable"] = [0, snap.mass]
        attributes(header)["BoxSize"] = snap.box_size

        parameters = create_group(f, "Parameters")
        attributes(parameters)["UnitLength_in_cm"] = snap.unit_length_cgs
        attributes(parameters)["UnitMass_in_g"] = snap.unit_mass_cgs
        attributes(parameters)["UnitVelocity_in_cm_per_s"] = snap.unit_vel_cgs

        parttype1 = create_group(f, "PartType1")
        parttype1["ParticleIDs"] = collect(1:length(snap.coords))
        parttype1["Coordinates"] = Array{Float64}(undef, 3, length(snap.coords))
        for (i,c) in enumerate(snap.coords)
            parttype1["Coordinates"][:, i] = [c.dx, c.dy, c.dz]
        end
        attributes(parttype1["Coordinates"])["to_cgs"] = 1.0
        attributes(parttype1["Coordinates"])["a_scaling"] = 1.0
        attributes(parttype1["Coordinates"])["h_scaling"] = -1.0
        attributes(parttype1["Coordinates"])["length_scaling"] = 1.0
        attributes(parttype1["Coordinates"])["mass_scaling"] = 0.0
    end

    # TODO: wouldn't it have been nice if I had stored the velocities
    # in the snapshot? :(
end

end # module