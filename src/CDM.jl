module CDM

include("geometry.jl")
include("snapshots.jl")
include("simulation.jl")

# REMINDER: run with -t 2 to avoid blocking simulation while writing snapshot
# to disk

ic = Snapshots.read_snap("./vendor/gadget4/output/snapdir_000/snapshot_ics_000.0.hdf5")
@time Simulation.run(ic, 
                     [0.041667f0, 0.111111f0,   0.166667f0, 0.25f0, 0.333333f0, 1.0f0],
#                     z ~= 23     z ~= 8        z ~= 5      z = 3   z ~= 2      z = 1.0
                     s -> Snapshots.write_snap(s, "./output/snapshot_z=$(replace(string(1/s.time - 1), '.' => '_'))_000.hdf5"),
                     0.3f0, # epsilon
                     0.4f0, # timestep crit accuracy
                     0.5f0, # bh opening crit
                     0.001f0, # max_time_step
                     64, # mesh size
                     100*1e5/(3.0857f0*1e24), # hubble's constant (1*100 km/s/Mpc) in CGS units
                     0.308f0, # matter density
                     0.0f0, # curvature of the universe
                     0.692f0, # omega lambda (fraction of the universe that is dark energy)
                     Float32(1e-5)) # time step

end # module