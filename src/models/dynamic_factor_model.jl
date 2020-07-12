"""
    dynamic_factor_model(y, k, beta_prior = 0, beta_var_prior = 1, gamma_prior = 1.5, delta_prior = 0.5, theta_prior = 0, theta_var_prior = 1; ndraw = 1500, burnin = 500, constant = true, sigma_squared0 = Nothing, display = true, display_step = 250)

Estimating bayesian dynamic linear factor model with normally distributed errors and k number of factors by sampling the
factors, the \$\\beta\$ factor loadings, the error variances and the autoregressive coefficients of te factor equation
with independent \$N(\\beta_{prior},V_{prior})\$, \$\\Gamma(\\gamma_{prior},\\delta_{prior})\$ and
\$N(\\theta_{prior},V_{\\theta,prior})\$ priors via Gibbs sampling returning an `ndraw-burnin` long sample.

\$y_t = f_t \\beta_t' + \\epsilon_t \\quad \\quad \\epsilon_{t,i} \\sim N(0,\\sigma^2_i)\$

\$f_{t,j} = \\theta_j f_{t,j-1} +\\eta_{t,j} \\quad \\quad \\eta_{t,j} \\sim N(0,1)\$

where \$i = 1...m\$, \$j = 1...k\$.

### The Gibbs sampling iterates on the following steps:
- sampling \$\\beta\$ factor loadings - see [`sampling_factor_loading`](@ref)
- sampling error variance - see [`sampling_sigma_squared`](@ref)
- sampling factor autoregressive coefficients
- sampling dynamic factors - see [`sampling_factor_dynamic`](@ref)

## Arguments
- `y::Vector`: (T x m) matrix of observations
- `k::Int`: number of factors to estimate
- `beta_prior::Number::Vector = 0`: () or (k) mean of the prior distribution of factor loadings (\$\\beta_{prior}\$)
- `beta_var_prior::Number::Vector::Matrix = 1`: () or (k) or (k x k) covariance of the prior distribution (\$V_{prior}\$)
- `gamma_prior::Number = 1.5`: shape of the prior distribution (\$\\gamma_{prior}\$)
- `delta_prior::Number = 0.5`: scale of the prior distribution (\$\\delta_{prior}\$)
- `theta_prior::Number = 0`: mean of the prior distribution of factor autoregressive coefficients (\$\\theta_{prior}\$)
- `theta_var_prior::Number = 1`: variance of the prior distribution of factor autoregressive coefficients (\$V_{\\theta,prior}\$)
- `ndraw::Int = 1500`: number of MCMC draws
- `burnin::Int = 500`: length of the burn-in period
- `sigma_squared0::Number = Nothing`: initial value of the error variance, if `Nothing` the initial value is `(gamma_prior - 1) / delta_prior`
- `display::Bool = true`: show model estimation progress
- `display_step::Int = 250`: index of every \$n\$th MCMC step is displayed

## Returns
- `sampled_beta::Matrix`: (m x k x ndraw-burnin) sampled \$\\beta\$ factor loadings
- `sampled_sigma_squared::Matrix`: (m x ndraw-burnin) sampled error variances
- `sampled_theta::Matrix`: (k x ndraw-burnin) sampled factor autoregressive coefficients
- `sampled_factor::Matrix`: (T x k x ndraw-burnin) sampled factors

Note: the prior mean of the error variance distribution is \$\\frac{\\gamma_{prior} - 1}{\\delta_{prior}}\$ (default = 1).
"""
function dynamic_factor_model(y, 
                              _k, 
                              _beta_prior = 0, 
                              _beta_var_prior = 1, 
                              _gamma_prior = 1.5, 
                              _delta_prior = 0.5,
                              _theta_prior = 0, 
                              _theta_var_prior = 1; 
                              ndraw = 1500, 
                              burnin = 500, 
                              sigma_squared0 = Nothing,
                              display = true, 
                              display_step = 250, 
                              k = _k, beta_prior = _beta_prior, beta_var_prior = _beta_var_prior, gamma_prior = _gamma_prior, 
                              delta_prior = _delta_prior, theta_prior = _theta_prior, theta_var_prior = _theta_var_prior)
                              
    T = size(y, 1)
    m = size(y, 2)

    # Create containers:
    sampled_beta = zeros(m, k, ndraw - burnin)
    sampled_sigma_squared = zeros(m, ndraw - burnin)
    sampled_theta = zeros(k, ndraw - burnin)
    sampled_factor = zeros(T, k, ndraw - burnin)

    # Initial values:
    sigma_squared = sigma_squared0 == Nothing ? ones(m) * (gamma_prior - 1) / delta_prior : sigma_squared0
    theta = zeros(k)
    factor = factor_initialize(y, k)

    # Sampling:
        display ? println("Estimate bayesian dynamic linear factor model (via Gibbs sampling)") : -1
        for i = 1:ndraw
            (mod(i, display_step) == 0 && display) ? println(i) : -1

            # Sampling:
            beta = sampling_factor_loading(y, factor, beta_prior, beta_var_prior, Diagonal(sigma_squared))  # sampling factor loadings
            sigma_squared = sampling_sigma_squared(y - factor * beta', gamma_prior, delta_prior)  # sampling error variances
            for j = 1:k
                theta[j] = sampling_beta(factor[2:T,j],factor[1:T-1,j],theta_prior,theta_var_prior,1,stationarity_check=true)[1]  # sampling factor AR(1) coefficients
            end
            factor = sampling_factor_dynamic(y, beta, theta, Diagonal(sigma_squared))  # sampling factors

            # Save samples:
            if i > burnin
                sampled_beta[:, :, i-burnin] = beta
                sampled_sigma_squared[:, i-burnin] = sigma_squared
                sampled_theta[:, i-burnin] = theta
                sampled_factor[:, :, i-burnin] = factor
            end
        end
        display ? println("Done") : -1
    return [sampled_beta, sampled_sigma_squared, sampled_theta, sampled_factor]
end
