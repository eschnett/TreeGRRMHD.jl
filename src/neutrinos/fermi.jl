# The complete Fermi–Dirac integrals of integer order, to round-off.
#
#     F_k(η) = ∫₀^∞ xᵏ dx / (exp(x − η) + 1),   k = 0 … 5
#
# The leakage rates read them at the electrons' and the neutrinos'
# degeneracies ("Neutrinos", "Leakage" in `CODE.md`). Not by the closed-form
# fits of Takahashi, El Eid & Hillebrandt (1978) that leakage codes usually
# take, but exactly (decided in step 16, the "(open)" of the Fermi integrals'
# fits): the same cost — one `exp` and a fixed loop — and no fit error to
# carry into a test.
#
# For η ≤ 0, F_k(η) = k! Σ_{j≥0} (−1)ʲ e^{(j+1)η} / (j+1)^{k+1}, an
# alternating series whose terms are the moments of a positive measure on
# [0, e^η] ⊆ [0, 1], so Cohen, Rodriguez Villegas & Zagier's (2000)
# Algorithm 1 sums it with relative error below 4/(3 + √8)ⁿ in n terms. For
# η > 0 the reflection formula for integer k,
#
#     F_k(η) = (−1)ᵏ F_k(−η) + P_k(η),
#     P_k(η) = Σ_{j=0}^{⌊(k+1)/2⌋} 2 (1 − 2^{1−2j}) ζ(2j) k!/(k+1−2j)! η^{k+1−2j},
#
# (ζ(0) = −½, so the leading term is η^{k+1}/(k+1)) takes it back to η ≤ 0.
# P_k(0) = 2 F_k(0) > F_k(0) for odd k, so the subtraction there loses
# nothing.

# The number of terms: 4/(3 + √8)ⁿ ≤ eps(T)/2, 22 at Float64 and 11 at
# Float32. A function of the type only, so the loop's trip count is a
# constant in a kernel.
@inline fermi_terms(::Type{T}) where {T} =
    ceil(Int, log(8 / Float64(eps(T))) / log(3 + sqrt(8.0)))

# CRVZ's Algorithm 1 sums Σ_{j≥0} (−1)ʲ a_j as Σ_{j<n} c_j a_j / d; with
# a_j = x^{j+1}/(j+1)^s that is a polynomial in x whose weights
# c_j/(d (j+1)^s) depend only on T and s. They are computed once, in
# BigFloat, when the method is compiled, so that a call is one unrolled
# Horner evaluation (measured in step 16: 117 ns a call with the recurrence
# in the loop).
@generated function crvz_weights(::Type{T}, ::Val{s}) where {T,s}
    n = fermi_terms(T)
    w = setprecision(BigFloat, 256) do
        d = (3 + sqrt(big(8)))^n
        d = (d + inv(d)) / 2
        b = -one(BigFloat)
        c = -d
        ws = BigFloat[]
        for j in 0:(n - 1)
            c = b - c
            push!(ws, c / (d * big(j + 1)^s))
            b = big((j + n) * (j - n)) * b / ((big(j) + big(1) / 2) * big(j + 1))
        end
        ws
    end
    return Tuple(T.(w))
end

# Σ_{j≥0} (−1)ʲ x^{j+1}/(j+1)^s for 0 ≤ x ≤ 1.
@inline alternating_polylog(x::T, ::Val{s}) where {T,s} =
    x * evalpoly(x, crvz_weights(T, Val(s)))

# F_k(η) for η ≤ 0.
@inline fermi_negative(::Val{k}, η::T) where {k,T} =
    T(factorial(k)) * alternating_polylog(exp(η), Val(k + 1))

# P_k(η) by Horner's rule; the coefficients are those of the comment above.
@inline function fermi_polynomial(::Val{k}, η::T) where {k,T}
    π² = T(π)^2
    z2 = π² / 6                   # 2(1 − 2⁻¹) ζ(2)
    z4 = 7 * π²^2 / 360           # 2(1 − 2⁻³) ζ(4)
    z6 = 31 * π²^3 / 15120        # 2(1 − 2⁻⁵) ζ(6)
    η² = η * η
    if k == 0
        return η
    elseif k == 1
        return η² / 2 + z2
    elseif k == 2
        return η * (η² / 3 + 2 * z2)
    elseif k == 3
        return η² * (η² / 4 + 3 * z2) + 6 * z4
    elseif k == 4
        return η * (η² * (η² / 5 + 4 * z2) + 24 * z4)
    else # k == 5
        return η² * (η² * (η² / 6 + 5 * z2) + 60 * z4) + 120 * z6
    end
end

# F_k(η) e^{−η} for η ≤ 0, which never underflows: k! times the series
# without its common factor e^η.
@inline fermi_scaled_negative(::Val{k}, η::T) where {k,T} =
    T(factorial(k)) * evalpoly(exp(η), crvz_weights(T, Val(k + 1)))

"""
    fermi_ratio(Val(a), Val(b), η)

`F_a(η)/F_b(η)`, finite for every finite `η`: below zero the common factor
`e^η` is taken out of both, so that a degeneracy of −1000, where each
integral underflows, still gives the ratio's limit `a!/b!`. The rates' mean
energies and spectral averages are such ratios.
"""
@inline function fermi_ratio(::Val{a}, ::Val{b}, η::T) where {a,b,T<:AbstractFloat}
    η ≤ 0 && return fermi_scaled_negative(Val(a), η) / fermi_scaled_negative(Val(b), η)
    return fermi(Val(a), η) / fermi(Val(b), η)
end

"""
    fermi(Val(k), η)

The complete Fermi–Dirac integral `F_k(η) = ∫₀^∞ xᵏ dx / (exp(x − η) + 1)` of
integer order `k = 0 … 5`, to a few ulp in `η`'s type: an accelerated
alternating series for `η ≤ 0` and the reflection formula above it
("Neutrinos", "Leakage" in `CODE.md`). Allocation-free and `isbits` in, so
callable in a kernel.
"""
@inline function fermi(::Val{k}, η::T) where {k,T<:AbstractFloat}
    0 ≤ k ≤ 5 || throw(ArgumentError(
        "fermi is defined for the orders 0 … 5 the leakage rates use, got $k."))
    η ≤ 0 && return fermi_negative(Val(k), η)
    tail = fermi_negative(Val(k), -η)
    return fermi_polynomial(Val(k), η) + (iseven(k) ? tail : -tail)
end
