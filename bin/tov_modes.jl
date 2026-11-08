# Step 10 (rung 4): the radial mode frequencies of the Cowling TOV star, from the central
# density's time series that `bin/tov_study.jl` writes (`out=…`), against Font et al. (2002,
# PRD 65, 084024, Table I: the fixed-spacetime frequencies of this model from an independent
# 2D code, F = 2.701, H1 = 4.563, H2 = 6.352 kHz).
#
#     julia --project=. bin/tov_modes.jl SERIES [t_skip=T] [t_end=T] [degree=K]
#
# The series' polynomial trend of degree `degree` is removed (1: the mean and a linear
# drift; higher for a run whose central density decays), the rest is Hann-windowed, and its discrete
# Fourier transform is evaluated on a fine grid of frequencies; each mode is the largest peak
# within ±20% of its reference, refined by a parabola through the peak and its neighbours;
# the peak is the largest local maximum inside the window, and a window with none (a flank
# rising to its edge) is reported as no peak (`NaN`).
# `t_skip` drops the start (the initial transient), `t_end` the rest after it (to compare
# runs over the same span). Frequencies in kHz: one `M☉` of time is
# `G M☉/c³ = 4.925490947 µs`.

using Printf

const MSUN_TIME_MS = 4.925490947e-3
const FONT2002_COWLING = (; F=2.701, H1=4.563, H2=6.352)

function read_series(path)
    t, ρ = Float64[], Float64[]
    for line in eachline(path)
        f = split(line)
        length(f) ≥ 2 || continue
        push!(t, parse(Float64, f[1]))
        push!(ρ, parse(Float64, f[2]))
    end
    return t, ρ
end

function spectrum(t, y, fs; degree=1)
    n = length(t)
    τ = (t .- t[1]) ./ (t[end] - t[1])
    A = hcat((τ .^ k for k in 0:degree)...)
    y = y - A * (A \ y)                              # the polynomial trend removed
    w = [sin(π * (i - 1) / (n - 1))^2 for i in 1:n]  # Hann
    dt = (t[end] - t[1]) / (n - 1)
    return [abs(sum(w[i] * y[i] * cis(-2π * f * t[i]) for i in 1:n)) * dt for f in fs]
end

function peak(fs, P, lo, hi)
    idx = findall(f -> lo ≤ f ≤ hi, fs)
    # Only interior local maxima: a value rising to the window's edge is the flank of a
    # peak outside it (H1's, in H2's window), not a mode.
    loc = filter(i -> i ∉ (first(idx), last(idx)) && P[i - 1] < P[i] > P[i + 1], idx)
    isempty(loc) && return NaN, maximum(P[idx])
    i = loc[argmax(P[loc])]
    a, b, c = P[i - 1], P[i], P[i + 1]
    δ = (a - c) / (2 * (a - 2b + c))                 # parabola through the three
    return fs[i] + δ * (fs[2] - fs[1]), b
end

function modes(path; t_skip=0.0, t_end=Inf, degree=1)
    t, ρ = read_series(path)
    keep = t_skip .≤ t .≤ t_end
    t, ρ = t[keep], ρ[keep]
    f_kHz(f) = f / MSUN_TIME_MS                      # 1/M☉ to kHz (per ms)
    fs = range(0.5, 8.0; length=7501) .* MSUN_TIME_MS   # kHz grid, in 1/M☉
    P = spectrum(t, ρ, fs; degree=round(Int, degree))
    @printf("%s: %d samples, t = %.1f … %.1f M (%.2f ms), resolution %.3f kHz\n", path,
            length(t), t[1], t[end], (t[end] - t[1]) * MSUN_TIME_MS,
            1 / ((t[end] - t[1]) * MSUN_TIME_MS))
    out = Dict{Symbol,Float64}()
    for (name, ref) in pairs(FONT2002_COWLING)
        f, p = peak(fs, P, 0.8 * ref * MSUN_TIME_MS, 1.2 * ref * MSUN_TIME_MS)
        out[name] = f_kHz(f)
        @printf("  %-2s %.3f kHz (Font et al. 2002: %.3f), %+.2f%%, peak %.3g\n", name,
                f_kHz(f), ref, 100 * (f_kHz(f) - ref) / ref, p)
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    path = ARGS[1]
    kw = Dict(Symbol(k) => parse(Float64, v) for (k, v) in (split(a, '=') for a in ARGS[2:end]))
    modes(path; kw...)
end
