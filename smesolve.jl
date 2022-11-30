using LinearAlgebra
using ProgressMeter
using SparseArrays

Operator = AbstractMatrix{<:Complex}

"""Solve the diffusive SME and return the real-valued detectors signal for
many trajectories.

Arguments:
- H: system Hamiltonian
- Ls: vector with operators characterising purely dissipative process
- Os: vector with operators characterising detectors
- etas: efficiency of the detectors
- rho0: initial quantum state
- T: total time of the stochastic trajectory
- Tbin: time of a single bin
- dt: numerical scheme discretization time
- ntraj: number of trajectories

Returns:
An array of dimension (number of trajectories, number of detectors, number of
bins) containing the real-valued signals.

See the following paper for the numerical method used:
    Pierre Rouchon and Jason F. Ralpha (2015)
    Efficient Quantum Filtering for Quantum Feedback Control
    Physical Review A, 91(1), 012118
    arXiv:1410.5345 [quant-ph]: https://arxiv.org/pdf/1410.5345.pdf

    Difference with the paper:
    - `V_j` is renamed `L_mu`
    - `L_i` is renamed `O_mu`
    - the last term of eq.(7) is ignored (it is two order of magnitude smaller
      than the other terms, as explained in the section VI. of the paper)
"""
function smesolve(
    H::Operator,
    Ls::AbstractVector{<:Operator},
    Os::AbstractVector{<:Operator},
    etas::Vector{<:Real},
    rho0::Operator;
    T::Real,
    Tbin::Real,
    dt::Real,
    ntraj::Int,
)::Array{<:Real, 3}  # (ntraj, nmeas, nbin)
    nbin = ceil(Int64, T / Tbin)
    nmeas = length(Os)

    trajs = zeros(Float64, ntraj, nmeas, nbin)

    p = Progress(ntraj, output=stdout)
    for i in 1:ntraj
        trajs[i, :, :] = _smesolve(H, Ls, Os, etas, rho0,T=T, Tbin=Tbin, dt=dt)
        next!(p)
    end

    return trajs
end

function _smesolve(
    H::Operator,
    Ls::AbstractVector{<:Operator},
    Os::AbstractVector{<:Operator},
    etas::Vector{<:Real},
    rho0::Operator;
    T::Real,
    Tbin::Real,
    dt::Real,
)::Matrix{<:Real}  # (nmeas, nbin)
    npoints = ceil(Int64, T / dt)
    nbin = ceil(Int64, T / Tbin)
    npointsperbin = ceil(Int64, npoints / nbin)
    nmeas = length(Os)
    Lsdag = adjoint.(Ls)
    Osdag = adjoint.(Os)
    setas = sqrt.(etas)
    sdt = sqrt(dt)
    rho = rho0

    ys = zeros(Float64, nmeas, nbin)
    dys = zeros(Float64, nmeas)
    ybins = zeros(Float64, nmeas)

    N = size(rho0)[1]
    zero = spzeros(ComplexF64, N, N)
    # define the sum of an empty vector of `Operator` to be the `zero`` operator
    sumz(As::AbstractVector{<:Operator}) = sum(As, init=zero)

    # M0 = I - iH dt - [sum(L'L) + sum(O'O)] dt / 2
    M0 = I - (1im * H + 0.5 * sumz(Lsdag .* Ls) + 0.5 * sumz(Osdag .* Os)) * dt

    for i in 1:npoints
        dWs = randn(nmeas) * sdt

        # dy_mu = sqrt(η_mu) tr[(O_mu + O_mu') ρ] dt + dW_mu
        dys = setas .* real.(tr.((Os .+ Osdag) .* [rho])) * dt .+ dWs

        # M = M0 + sum(sqrt(η_mu) O_mu dy_mu)
        M = M0 + sumz(setas .* Os .* dys)

        # ρ = M ρ M' + sum(L ρ L') * dt + sum[(1 - η_mu) O_mu ρ O_mu'] * dt
        rho = M * rho * adjoint(M) +
                sumz(Ls .* [rho] .* Lsdag) * dt +
                sumz((1 .- etas) .* Os .* [rho] .* Osdag) * dt

        # ρ = ρ / tr(ρ)
        rho = rho / tr(rho)

        # integrate the signal in the bin
        ybins += dys

        # store the integrated signal if end of bin
        if i % npointsperbin == 0
            bin_idx = floor(Int64, i / npointsperbin)
            ys[:, bin_idx] = ybins
            ybins = fill(0.0, nmeas)
        end
    end

    return ys / Tbin
end
