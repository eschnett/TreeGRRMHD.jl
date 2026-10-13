# The operations this package performs that a *software* floating-point type
# does not provide, written so that they need only what every type here has.
#
# The package is generic in its element type `T`, and TreeAMR's mesh is too
# (see "Precision and devices" in `CODE.md`). What is not generic is `Base`:
# MultiFloats.jl implements `floor` and `ceil` returning a float, but no
# conversion to `Integer` and no `rem`, so `ceil(Int, x)` and `mod(x, y)` --
# both of which close through those -- are `MethodError`s at `Float32x2` while
# working at `Float32` and `Float64`. Neither is a physics decision, so neither
# belongs spelled out at the call site.
#
# Copied from TreeHydro (which copied it from TreeWave) rather than depended
# on: the packages are read together, but a physics package does not depend on
# a sample application.
#
# Nothing here is exported. They are spellings, not concepts.

"""
    wrap(x, L)

`x` reduced into `[0, L)` for positive `L` — what `mod(x, L)` means on a
periodic box.

Spelled `x - L·floor(x/L)` rather than `mod`, because `Base.mod` on floats
goes through `rem`, which MultiFloats.jl does not define. The two agree
wherever `L > 0`, which is the only case a box produces; at `L = 1` both are
exact and the results are bit-identical.

Its consumers will be the periodic flat-space cases of rung 1 and 2 — the
vacuum electromagnetic wave and the circularly polarized Alfvén wave — whose
exact solutions are evaluated at `x - v t` and have to be brought back into
the box before they are compared with the run.
"""
wrap(x, L) = x - L * floor(x / L)

"""
    ceilint(x)
    floorint(x)
    roundint(x)

`ceil(Int, x)`, `floor(Int, x)` and `round(Int, x)`, for a type that may not
define `Int(::AbstractFloat)`.

Both `ceil(Int, x)` and `floor(Int, x)` close through a conversion to
`Integer` that MultiFloats.jl does not provide — and neither does a detour
through `Float64`, since `Float64(::Float32x2)` is not defined either. What
*every* `AbstractFloat` in Julia converts to is `BigFloat`, so that is the
fallback, taken only once the value is already an exact integer and therefore
only ever exact. A hardware float never reaches it.

The fallback allocates, which is why these are confined to host-side control
flow evaluated a handful of times per run — a count of chunks, of steps in a
chunk, of buffer cells, a sum of floor flags closed to an `Int` — and never
anything inside a kernel or per cell.
"""
ceilint(x) = _toint(ceil(x))
floorint(x) = _toint(floor(x))
roundint(x) = _toint(round(x))

# `y` is an exact integer value by construction, so both branches are exact.
_toint(y::Base.IEEEFloat) = Int(y)
_toint(y) = Int(BigFloat(y))

"""
    tofloat64(x)

`x` as a `Float64` — the bridge to host code that computes in `Float64`
whatever type the run itself uses: reference solutions, the host TOV solve,
and the record a run hands back.

`Float64(x)` is not universal either: MultiFloats.jl defines a conversion
only to its own *limb* type, so `Float64(::Float32x2)` is a `MethodError`
while `Float32(::Float32x2)` is not. `BigFloat` is again the common currency.
Host-side, and a handful of times per run, never per cell.
"""
tofloat64(x::Base.IEEEFloat) = Float64(x)
tofloat64(x) = Float64(BigFloat(x))
