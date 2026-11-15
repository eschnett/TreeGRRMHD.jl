# Step 11 (rung 5): the overtones of a perturbed TOV star, from two central-density series
# on the same mesh, one with `δv` (`tov_case`'s eigenfunction perturbation) and one without
# ("Validation ladder" 5, "The perturbed star" in `CODE.md`). The difference
# `Δρ_c/ρ_c = (ρ_c(δv) − ρ_c(0))/ρ_c(0)` cancels the unperturbed run's own lines (the
# truncation-excited modes and the spacetime's constraint lines), and what remains is fitted
# by the matrix-pencil method (Hua & Sarkar 1990) to a sum of damped sinusoids, at several
# model orders: a mode that decays within a few periods has no sharp line in a windowed
# spectrum (`bin/tov_modes.jl`), but it has a pole. Per order, each pole's frequency in kHz,
# its e-folding time in M and its amplitude, with the fit's relative residual; and the
# envelope, the largest `|Δρ_c/ρ_c|` per window.
#
#     julia --project=. bin/tov_overtones.jl SERIES_δv SERIES_0 [t_skip=3] [t_end=T]
#           [orders=6,8,10] [window=25] [amin=0]
#
# Both files are `bin/tov_study.jl`'s or `coupled/tov_coupled_study.jl`'s (`t ρ_c …` per
# line), sampled at the same times; the longer one is cut to the shorter. `t_skip` drops
# the start, `t_end` the end; `amin` hides poles below that amplitude. Uniform sampling is
# required (the studies' `chunk`).

using LinearAlgebra
using Printf

const MSUN_TIME_MS = 4.925490947e-3                  # `G M☉/c³` in ms

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

# The matrix pencil of order `M`: the `M` poles `zₖ` of the Hankel matrix's dominant
# right singular subspace, `y(n) ≈ Σ aₖ zₖⁿ`, and the amplitudes by least squares.
function pencil(y, dt, M)
    N = length(y)
    L = N ÷ 3
    Y = [y[i + j - 1] for i in 1:(N - L), j in 1:(L + 1)]
    V = svd(Y).V[:, 1:M]
    z = eigvals(pinv(V[1:(end - 1), :]) * V[2:end, :])
    Z = [zk^(n - 1) for n in 1:N, zk in z]
    a = Z \ complex.(y)
    return log.(z) ./ dt, a, norm(real(Z * a) - y) / norm(y)
end

function main(args)
    length(args) ≥ 2 || error("tov_overtones.jl: give the two series, with δv and without")
    t_skip, t_end, orders, window, amin = 3.0, Inf, (6, 8, 10), 25.0, 0.0
    for a in args[3:end]
        k, v = split(a, '='; limit=2)
        if k == "t_skip"
            t_skip = parse(Float64, v)
        elseif k == "t_end"
            t_end = parse(Float64, v)
        elseif k == "orders"
            orders = Tuple(parse.(Int, split(v, ',')))
        elseif k == "window"
            window = parse(Float64, v)
        elseif k == "amin"
            amin = parse(Float64, v)
        else
            error("tov_overtones.jl: unknown key `$k`")
        end
    end
    t, a = read_series(args[1])
    t0, b = read_series(args[2])
    m = min(length(t), length(t0))
    t[1:m] == t0[1:m] || error("tov_overtones.jl: the two series are sampled at different times")
    d = (a[1:m] .- b[1:m]) ./ b[1:m]
    t = t[1:m]
    @printf("%s − %s: t = %g … %g M\n", args[1], args[2], t[1], t[end])
    print("  max |Δρ_c/ρ_c| per $(window) M:")
    for lo in t[1]:window:(t[end] - window)
        @printf(" %.2e", maximum(abs, d[lo .≤ t .< lo + window]))
    end
    println()
    k = t_skip .≤ t .≤ t_end
    tk, dk = t[k], d[k]
    dt = tk[2] - tk[1]
    all(x -> isapprox(x, dt; rtol=1e-9), diff(tk)) ||
        error("tov_overtones.jl: the series is not uniformly sampled")
    for M in orders
        s, c, r = pencil(dk, dt, M)
        @printf("  order %2d over [%g, %g] M, residual %.2e:\n", M, tk[1], tk[end], r)
        rows = sort([(imag(s[i]) / 2π / MSUN_TIME_MS, -1 / real(s[i]), 2 * abs(c[i]))
                     for i in eachindex(s) if imag(s[i]) > 0]; by=first)
        for (f, τ, A) in rows
            A ≥ amin && @printf("    %7.3f kHz  τ %9.1f M  amplitude %.2e\n", f, τ, A)
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
