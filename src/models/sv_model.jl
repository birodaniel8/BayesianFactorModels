"""
    sv_model(y, rho_prior = 0, rho_var_prior = 10, gamma_tau_prior = 1.5, delta_tau_prior = 0.05, ndraw=1000, burnin=500, tau0 = delta_tau_prior/(gamma_tau_prior-1), h0 = 0, P0 = 10)

Estimating bayesian stochastic volatility model by sampling the static (\$\\sigma^2\$) and stochastic (\$h_t\$)
volatility components and the volatility process parameters with independent normal and \$\\Gamma\$ priors on the
autoregressive process parameters and the error variances via Gibbs sampling returning an `ndraw-burnin` long sample.

\$y_t = e^{h_t/2} \\epsilon_t  \\quad \\quad \\epsilon_i \\sim N(0,1)\$

\$h_t = \\rho_0 + \\rho_1 h_{t-1} + \\eta_t \\quad \\quad \\eta_t \\sim N(0,\\tau^2)\$

### The Gibbs sampling iterates on the following steps:
- sampling stochastic volatility process parameters with stationarity conditions - see [`sampling_beta`](@ref)
- sampling the volatility of stochastic volatility - see [`sampling_sigma_squared`](@ref)
- sampling the random walk stochastic volatility (Kim et al 1998) - see [`sampling_sv_rw`](@ref)

## Arguments
- `y::Vector`: (T) vector of observations
- `rho_prior::Vector::Number = 0`: prior mean of the stochatis volatility process parameters
- `rho_var_prior::Vector::Number = 10`: prior variance of the stochatis volatility process parameters
- `gamma_tau_prior::Number = 1.5`: shape of the prior distribution of the variance of the stochastic volatility
- `delta_tau_prior::Number = 0.05`: scale of the prior distribution of the variance of the stochastic volatility
- `ndraw::Int = 1500`: number of MCMC draws
- `burnin::Int = 500`: length of the burn-in period
- `tau0::Number = delta_tau_prior/(gamma_tau_prior-1)`: initial value of the volatility of stochastic volatility
- `h0:Number = 0`: initial state value of the stochastic volatility
- `P0:Number = 1`: initial state variance of the stochastic volatility

## Returns
- `sampled_rho::Matrix`: (2 x ndraw-burnin) sampled stochastic volatility process parameters
- `sampled_tau::Vector`: (ndraw-burnin) sampled volatility of volatilities
- `sampled_factor::Matrix`: (T x ndraw-burnin) sampled stochastic volatility

Note: We use the "variance" and "volatility" terms in the description, however all the priors and the sampled values are on
the variance term!
"""
function sv_model(y, 
                  _rho_prior = 0, 
                  _rho_var_prior = 10, 
                  _gamma_tau_prior = 1.5, 
                  _delta_tau_prior = 0.05, 
                  _ndraw=1000,
                  _burnin=500, 
                  _tau0 = _delta_tau_prior/(_gamma_tau_prior-1), 
                  _h0 = 0, 
                  _P0 = 10;
                  rho_prior = _rho_prior, rho_var_prior = _rho_var_prior, gamma_tau_prior = _gamma_tau_prior,
                  delta_tau_prior = _delta_tau_prior, ndraw = _ndraw, burnin = _burnin, tau0 = _tau0, h0 = _h0, P0 = _P0)
                  
    T = size(y, 1)

    # Create containers:
    sampled_rho = zeros(2, ndraw-burnin)
    sampled_tau = zeros(ndraw-burnin)
    sampled_h = zeros(T, ndraw-burnin)

    # Initial values:
    h = log.(y.^2)
    tau = tau0

    for i = 1:ndraw
        # Sampling:
        (mod(i, 100) == 0 && true) ? println(i) : -1
        rho = sampling_beta(h[2:T],[ones(T-1) h[1:T-1]],sigma_squared = tau, beta_prior = rho_prior, beta_var_prior = rho_var_prior,
                            stationarity_check = true, constant_included=true)  # sampling the volatility process parameters
        tau = sampling_sigma_squared(h[2:T] .- rho[1] - rho[2] .* h[1:T-1], gamma_tau_prior, delta_tau_prior)  # sampling volatility of volatility
        h = sampling_sv(y, h, rho, tau, h0, P0)  # sampling stochastic volatilty

        # Save samples:
        if i > burnin
            sampled_rho[:,i-burnin] = rho
            sampled_tau[i-burnin] = tau[1]
            sampled_h[:,i-burnin] = h
        end
    end
    return [sampled_rho, sampled_tau,sampled_h]
end
