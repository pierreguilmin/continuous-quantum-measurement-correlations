using KrylovKit
using LinearAlgebra

Operator = AbstractMatrix{<:Complex}

"""Compute the action of the superoperator D[L](.) on a density matrix."""
function D(L::Operator, rho::Operator)::Operator
    Ldag = adjoint(L)
    return L * rho * Ldag - 0.5 * Ldag * L * rho - 0.5 * rho * Ldag * L
end

"""Compute the action of the Lindbladian on a density matrix."""
function lindbladian(
    H::Operator,
    Ls::AbstractVector{<:Operator},
    rho::Operator,
)::Operator
   return - 1im * (H * rho - rho * H) + sum(D(L, rho) for L in Ls)
end

"""Compute the action of the superoperator L+(.) on a density matrix."""
function Lplus(L::Operator, rho::Operator)::Operator
    return L * rho + rho * adjoint(L)
end

"""Compute the coupled ODEs generators. `Ls` is a vector with all the jump
operators and `idx` is the index of the operator characterising the detector."""
function odes_generator(
    H::Operator,
    Ls::AbstractVector{<:Operator},
    idx::Number,
    eta::Real,
    rhos::AbstractArray{<:Complex, 3},
    in_O1::Bool,
    in_O2::Bool,
)::AbstractArray{<:Complex, 3}
    rho, rho1, rho2, rho12 = rhos[1, :, :], rhos[2, :, :], rhos[3, :, :], rhos[4, :, :]
    L = Ls[idx]

    drho = lindbladian(H, Ls, rho)
    drho1 = lindbladian(H, Ls, rho1) + in_O1 * sqrt(eta) * Lplus(L, rho)
    drho2 = lindbladian(H, Ls, rho2) + in_O2 * sqrt(eta) * Lplus(L, rho)
    drho12 = (
        lindbladian(H, Ls, rho12)
        + in_O1 * sqrt(eta) * Lplus(L, rho2)
        + in_O2 * sqrt(eta) * Lplus(L, rho1)
        + in_O1 * in_O2 * rho
    )

    # some boilerplate code so `drhos` has the correct vector interface for
    # KrylovKit.jl, see https://github.com/Jutho/KrylovKit.jl/issues/63
    N = size(rho)[1]
    drhos = zeros(ComplexF64, 4, N, N)
    drhos[1, :, :] = drho
    drhos[2, :, :] = drho1
    drhos[3, :, :] = drho2
    drhos[4, :, :] = drho12

    return drhos
end

"""Compute the binned signal two-point correlation function by evolving the
coupled ODEs using Krylov subspace methods. `Ls` is a vector with all the jump
operators and `idx` is the index of the operator characterising the detector."""
function binned_theory(
    H::Operator,
    Ls::AbstractVector{<:Operator},
    idx::Number,
    eta::Real,
    rho0::Operator,
    O1::Tuple{Real, Real},
    O2::Tuple{Real, Real},
)::Real
    N = size(rho0)[1]

    # initial condition
    rhos = zeros(ComplexF64, 4, N, N)
    rhos[1, :, :] = rho0

    (a, b), (c, d) = O1, O2
    overlap = c <= b
    # no overlap: --- a --- b --- c --- d --->
    # overlap:    --- a --- c --- b --- d --->
    duration_before = a
    duration_O1 = overlap ? c - a : b - a
    duration_middle = overlap ? b - c : c - b
    duration_O2 = overlap ? d - b : d - c

    gen_before(rhos) = odes_generator(H, Ls, idx, eta, rhos, false, false)
    gen_O1(rhos) = odes_generator(H, Ls, idx, eta, rhos, true, false)
    gen_O2(rhos) = odes_generator(H, Ls, idx, eta, rhos, false, true)
    gen_middle(rhos) = odes_generator(H, Ls, idx, eta, rhos, overlap, overlap)

    # solve the ODEs on each piecewise constant part by computing the action
    # of the exponential of the generator on the state using KrylovKit.jl
    rhos = exponentiate(gen_before, duration_before, rhos)[1]
    rhos = exponentiate(gen_O1, duration_O1, rhos)[1]
    rhos = exponentiate(gen_middle, duration_middle, rhos)[1]
    rhos = exponentiate(gen_O2, duration_O2, rhos)[1]

    return real(tr(rhos[4, :, :]))
end
