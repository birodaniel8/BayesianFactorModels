using LinearAlgebra
using StatsBase
include(joinpath(pwd(), "sampling\\sampling_beta.jl"))
include(joinpath(pwd(), "sampling\\sampling_sigma_squared.jl"))
include(joinpath(pwd(), "sampling\\sampling_sv_rw.jl"))

"""
    sv_model_rw(y, gamma_prior = 1.5, delta_prior = mean(y.^2), gamma_tau_prior = 1.5, delta_tau_prior = 0.05, ndraw=1000, burnin=500, h0 = 0, P0 = 10)

Estimating bayesian random walk stochastic volatility model by sampling the static (\$\\sigma^2\$) and stochastic (\$h_t\$)
volatility components with independent \$\\Gamma\$ priors on the error variances via Gibbs sampling returning an
`ndraw-burnin` long sample.

\$y_t = \\sigma_t e^{h_t/2} \\epsilon_t  \\quad \\quad \\epsilon_i \\sim N(0,1)\$

\$h_t = h_{t-1} + \\eta_t \\quad \\quad \\eta_t \\sim N(0,\\tau^2)\$

### The Gibbs sampling iterates on the following steps:
- sampling \$\\sigma^2\$ static variance - see [`sampling_sigma_squared`](@ref)
- sampling the volatility of stochastic volatility - see [`sampling_sigma_squared`](@ref)
- sampling the random walk stochastic volatility (Kim et al 1998) - see [`sampling_sv_rw`](@ref)

## Arguments
- `y::Vector`: (T) vector of observations
- `gamma_prior::Number = 1.5`: shape of the prior distribution of the static variance
- `delta_prior::Number = mean(y.^2)`: scale of the prior distribution of the static variance
- `gamma_tau_prior::Number = 1.5`: shape of the prior distribution of the variance of the stochastic volatility
- `delta_tau_prior::Number = 0.05`: scale of the prior distribution of the variance of the stochastic volatility
- `ndraw::Int = 1500`: number of MCMC draws
- `burnin::Int = 500`: length of the burn-in period
- `h0:Number = 0`: initial state value of the stochastic volatility
- `P0:Number = 1`: initial state variance of the stochastic volatility

## Returns
- `sampled_sigma_squared::Vector`: (ndraw-burnin) sampled static variances
- `sampled_tau::Vector`: (ndraw-burnin) sampled volatility of volatilities
- `sampled_factor::Matrix`: (T x ndraw-burnin) sampled stochastic volatility

Note: We use the "variance" and "volatility" terms in the description, however all the priors and the sampled values are on
the variance term!
"""
function sv_model_rw(y, _gamma_prior = 1.5, _delta_prior = mean(y.^2), _gamma_tau_prior = 1.5, _delta_tau_prior = 0.05,
                     _ndraw=1000, _burnin=500, _h0 = 0, _P0 = 10;
                     gamma_prior = _gamma_prior, delta_prior = _delta_prior, gamma_tau_prior = _gamma_tau_prior,
                     delta_tau_prior = _delta_tau_prior, ndraw = _ndraw, burnin = _burnin, h0 = _h0, P0 = _P0)
    T = size(y, 1)

    # Create containers:
    sampled_sigma_squared = zeros(ndraw-burnin)
    sampled_tau = zeros(ndraw-burnin)
    sampled_h = zeros(T, ndraw-burnin)

    # Initial values:
    h = zeros(T)

    for i = 1:ndraw
        # Sampling:
        (mod(i, 100) == 0 && true) ? println(i) : -1
        sigma_squared = sampling_sigma_squared(y, gamma_prior, delta_prior)  # sampling static variance
        tau = sampling_sigma_squared(h[2:T] - h[1:T-1], gamma_tau_prior, delta_tau_prior)  # sampling volatility of volatility
        h = sampling_sv_rw(y, sigma_squared[1], h, tau[1], h0, P0)  # sampling stochastic volatilty
        h = h .- mean(h)  # demeaning h to 0

        # Save samples:
        if i > burnin
            sampled_sigma_squared[i - burnin] = sigma_squared[1]
            sampled_tau[i - burnin] = tau[1]
            sampled_h[:,i - burnin] = h
        end
    end
    return [sampled_sigma_squared, sampled_tau, sampled_h]
end
