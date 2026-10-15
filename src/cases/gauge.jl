# A pure gauge pulse (step 8): the test of the gauge damping `ξ` against the
# spurious `B` a refinement boundary makes ("Refinement" in `CODE.md`).
#
# Nothing physical moves: a uniform gas at rest, no field, no current
# (`Vacuum`), and the vector potential `A = 0` with a Gaussian in the gauge
# scalar, `Φ̃ = a exp(−r²/w²)`. The gauge equations then carry a pure gauge
# wave, `∂ₜA = −∇Φ̃`, `∂ₜΦ̃ = −∇·A − ξΦ̃` (flat space), and `A` stays a gradient:
# on a uniform mesh the discrete `−∇Φ̃` is a discrete gradient, whose discrete
# curl is zero to rounding, so `𝔅` stays zero to rounding for as long as the
# run lasts. Across a coarse-fine face it does not — the prolongation of a
# coarse discrete gradient is not a fine one — and the `𝔅` that appears is the
# refinement boundary's alone.

# p₁ … p₈ = a, w, ρ, p, —, —, —, Γ; s₁ the entropy.
@inline function wave_fields(w::MultiDCase{:gauge_pulse}, x::SVector{3,T}, t::T) where {T}
    z = zero(T)
    zv = SVector(z, z, z)
    return ideal_fields(w.p₃, zv, w.s₁, w.Yₑ, zv)
end

@inline function wave_potential(w::MultiDCase{:gauge_pulse}, x::SVector{3,T}, t::T) where {T}
    r² = x[1]^2 + x[2]^2 + x[3]^2
    return zero(SVector{3,T}), w.p₁ * exp(-r² / w.p₂^2)
end

"""
    gauge_pulse_case([T]; a = 1, w = 1//10, roots = 2)

A pure gauge pulse (proposed in step 8): a uniform Γ = 5/3 gas at rest with
`ρ = p = 1` ([`shock_gas`](@ref)), no field and no current (`Vacuum`), `A = 0`
and `Φ̃ = a exp(−r²/w²)`, on the periodic `[−1/2, 1/2]²`. The gauge
equations carry it as a pure gauge wave in which `A` stays a gradient, so `𝔅`
stays zero to rounding on a uniform mesh, and what `𝔅` appears on a refined
one is made at the refinement boundary — the measurement that decides the
gauge damping `ξ` ("Refinement" in `CODE.md`).
"""
function gauge_pulse_case(::Type{T}=Float64; a=1, w=1 // 10, roots=2) where {T}
    Γ = 5 // 3
    eos = shock_gas(T, Γ)
    s = EntropyEOS.polytropic_entropy(eos, one(T))
    p = MultiDCase{:gauge_pulse,T}(T(a), T(w), one(T), one(T), zero(T), zero(T), zero(T), T(Γ), s,
                                   s, T(1 // 2), false)
    h = T(1 // 2)
    return MHDCase(T, Val(2); fields=WaveFields(p), potential=WavePotential(p), eos=eos,
                   pol=EntropyEOS.default_policy(eos, T(1 // 10^10)), model=Vacuum(),
                   periodic=true, extents=((-h, h), (-h, h)), roots=roots)
end
