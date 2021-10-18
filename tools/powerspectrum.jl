using Plots, Statistics

# Read a GADGET4 power spectrum
function read_and_rebin_power_spectrum(inputps::T) where {T <: AbstractString}
    lines = readlines(inputps)

    nbins = parse(Int64, lines[2])
 
    ks = Array{Float64, 1}(undef, length(nbins))
    pws = Array{Float64, 1}(undef, length(nbins))

    # Read the power spectrum only for the case of unmodified periodic box
    for line in lines[6:6+nbins-1]
        parts = split(line, " ")

        k = parse(Float64, parts[1])
        #delta = parts[2]
        avgpw = parse(Float64, parts[3])
        #nomodes = parts[4]
        #snlimit = parts[5]

        push!(ks, k)
        push!(pws, avgpw)
    end

    # Rebin the spectrum to n = 100 bins
    rebin_samples = 100
    rebinned_ks = Array{Float64, 1}(undef, rebin_samples)
    rebinned_pws = Array{Float64, 1}(undef, rebin_samples)

    stride = ceil(Int64, length(ks)/rebin_samples)

    for i=1:stride:length(ks)
        mean_k = mean(ks[i:min(i+stride, length(ks))])
        mean_pw = mean(pws[i:min(i+stride, length(pws))])

        push!(rebinned_ks, mean_k)
        push!(rebinned_pws, mean_pw)
    end

    return (rebinned_ks, rebinned_pws)
end

function plot_compare_power_spectrums(inputps1, inputps2, outputplt, ps1label, ps2label)
    ks1, pws1 = read_and_rebin_power_spectrum(inputps1)
    ks2, pws2 = read_and_rebin_power_spectrum(inputps2)

    plot(ks1, pws1, label=ps1label)
    plot!(ks2, pws2, label=ps2label, dpi=1200)
    xlabel!("Número de onda k [h/Mpc]")
    ylabel!("P(k) [(h^-1 Mpc)^3]")

    savefig(outputplt)
end