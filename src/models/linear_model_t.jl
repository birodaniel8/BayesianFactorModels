"""
    linear_model_t(y, x, beta_prior = 0, beta_var_prior = 1, gamma_prior = 1.5, delta_prior = 0.5 v_prior = 30; ndraw = 1500, burnin = 500, constant = true, sigma_squared0 = Nothing, df0 = Nothing, mh_variance = 0.25, display = true, display_step = 250)

Estimating bayesian linear model with Student's t distributed errors. This model is equivalent to sampling from the normal
linear model with known heteroskedasticity. This function takes samples of the \$\\beta\$ coefficients, the error
variance and the degree of freedom parameter with independent \$N(\\beta_{prior},V_{prior})\$,
\$\\Gamma(\\gamma_{prior},\\delta_{prior})\$ and \$exp(v_{prior})\$ priors via Gibbs sampling returning an `ndraw-burnin`
long sample.

\$y = \\beta x + \\epsilon \\quad \\quad \\epsilon_i \\sim t(0,\\sigma^2,v)\$

### The Gibbs sampling iterates on the following steps:
- sampling \$\\beta\$ regression coefficients - see [`sampling_beta`](@ref)
- sampling error variance - see [`sampling_sigma_squared`](@ref)
- sampling the degfree of freedom parameter - see [`sampling_df`](@ref)
- sampling mixture scale parameters - [`sampling_mixtrue_scale`](@ref)

## Arguments
- `y::Vector`: (T x 1) dependent variable
- `x::Matrix`: (T x k) explanatory variables
- `beta_prior::Number::Vector = 0`: () or (k) mean of the prior distribution (\$\\beta_{prior}\$)
- `beta_var_prior::Number::Vector::Matrix = 1`: () or (k) or (k x k) covariance of the prior distribution (\$V_{prior}\$)
- `gamma_prior::Number = 1.5`: shape of the prior distribution (\$\\gamma_{prior}\$)
- `delta_prior::Number = 0.5`: scale of the prior distribution (\$\\delta_{prior}\$)
- `v_prior::Number = 30`: prior on the degree of freedom parameter (\$v_{prior}\$)
- `ndraw::Int = 1500`: number of MCMC draws
- `burnin::Int = 500`: length of the burn-in period
- `sigma_squared0::Number = Nothing`: initial value of the error variance, if `Nothing` the initial value is `(gamma_prior - 1) / delta_prior`
- `df0::Number = Nothing`: initial value of the degree of freedom, if `Nothing` the initial value is `v_prior`
- `display::Bool = true`: show model estimation progress
- `hm_variance::Number = 0.25`: random walk Metropolis-Hastings algorithm variance parameter

## Returns
- `sampled_beta::Matrix`: (k x ndraw-burnin) sampled \$\\beta\$ coefficients
- `sampled_sigma_squared::Matrix`: (1 x ndraw-burnin) sampled error variances
- `sampled_df::Matrix`: (1 x ndraw-burnin) sampled degree of freedom

Note: the prior mean of the error variance distribution is \$\\frac{\\gamma_{prior} - 1}{\\delta_{prior}}\$ (default = 1).
"""
function linear_model_t(y, x, _beta_prior = 0, _beta_var_prior = 1, _gamma_prior = 1.5, _delta_prior = 0.5, _v_prior = 30;
                        ndraw = 1500, burnin = 500, constant = true, sigma_squared0 = Nothing, df0 = Nothing,
                        hm_variance = 0.25, display = true, display_step = 250, beta_prior = _beta_prior,
                        beta_var_prior = _beta_var_prior, gamma_prior = _gamma_prior, delta_prior = _delta_prior,
                        v_prior = _v_prior)
    t = size(x, 1)
    k = size(x, 2)

    # Add constant:
    if constant
        x = [ones(t) x]
        k = k + 1
    end

    # Create containers:
    sampled_beta = zeros(k, ndraw - burnin)
    sampled_sigma_squared = zeros(1, ndraw - burnin)
    sampled_df = zeros(1, ndraw - burnin)

    # Initial values
    sigma_squared = sigma_squared0 == Nothing ? (gamma_prior - 1) / delta_prior : sigma_squared0
    lambda = ones(t)
    df = df0 == Nothing ? v_prior : df0

    # Sampling:
    display ? println("Estimate linear model with Student's t errors (via Gibbs sampling)") : -1
    for i = 1:ndraw
        (mod(i, display_step) == 0 && display) ? println(i) : -1

        # Sampling:
        y_star = Diagonal(sqrt.(lambda[:,1])) * y
        x_star = Diagonal(sqrt.(lambda[:,1])) * x
        beta = sampling_beta(y_star, x_star, beta_prior, beta_var_prior, sigma_squared)  # sampling beta coefficients
        sigma_squared = sampling_sigma_squared(y_star - x_star * beta[:,:], gamma_prior, delta_prior)  # sampling error variance
        df = sampling_df(lambda, df, v_prior, hm_variance)  # sampling degree of freedom
        lambda = sampling_mixtrue_scale(y - x * beta[:,:], sigma_squared, df)  # sampling lambda

        # Save samples:
        if i > burnin
            sampled_beta[:, i-burnin] = beta
            sampled_sigma_squared[:, i-burnin] = sigma_squared
            sampled_df[:, i-burnin] = df
        end
    end
    display ? println("Done") : -1
    return [sampled_beta, sampled_sigma_squared, sampled_df]
end
