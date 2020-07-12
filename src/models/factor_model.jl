"""
    factor_model(y, k, beta_prior = 0, beta_var_prior = 1, gamma_prior = 1.5, delta_prior = 0.5; ndraw = 1500, burnin = 500, constant = true, sigma_squared0 = Nothing, display = true, display_step = 250)

Estimating bayesian linear factor model with normally distributed errors and k number of factors by sampling the factors,
the \$\\beta\$ factor loadings and the error variances with independent \$N(\\beta_{prior},V_{prior})\$ and
\$\\Gamma(\\gamma_{prior},\\delta_{prior})\$ priors via Gibbs sampling returning an `ndraw-burnin` long sample.

\$y_t = f_t \\beta_t' + \\epsilon_t \\quad \\quad \\epsilon_{t,i} \\sim N(0,\\sigma^2_i)\$

\$f_{t,j} \\sim N(0,1)\$

where \$i = 1...m\$, \$j = 1...k\$.

### The Gibbs sampling iterates on the following steps:
- sampling \$\\beta\$ factor loadings - see [`sampling_factor_loading`](@ref)
- sampling error variance - see [`sampling_sigma_squared`](@ref)
- sampling factors - see [`sampling_factor`](@ref)

## Arguments
- `y::Vector`: (T x m) matrix of observations
- `k::Int`: number of factors to estimate
- `beta_prior::Number::Vector = 0`: () or (k) mean of the prior distribution of factor loadings (\$\\beta_{prior}\$)
- `beta_var_prior::Number::Vector::Matrix = 1`: () or (k) or (k x k) covariance of the prior distribution (\$V_{prior}\$)
- `gamma_prior::Number = 1.5`: shape of the prior distribution (\$\\gamma_{prior}\$)
- `delta_prior::Number = 0.5`: scale of the prior distribution (\$\\delta_{prior}\$)
- `ndraw::Int = 1500`: number of MCMC draws
- `burnin::Int = 500`: length of the burn-in period
- `sigma_squared0::Number = Nothing`: initial value of the error variance, if `Nothing` the initial value is `(gamma_prior - 1) / delta_prior`
- `display::Bool = true`: show model estimation progress
- `display_step::Int = 250`: index of every \$n\$th MCMC step is displayed

## Returns
- `sampled_beta::Matrix`: (m x k x ndraw-burnin) sampled \$\\beta\$ factor loadings
- `sampled_sigma_squared::Matrix`: (m x ndraw-burnin) sampled error variances
- `sampled_factor::Matrix`: (T x k x ndraw-burnin) sampled factors

Note: the prior mean of the error variance distribution is \$\\frac{\\gamma_{prior} - 1}{\\delta_{prior}}\$ (default = 1).
"""
function factor_model(y, 
                      _k, 
                      _beta_prior = 0, 
                      _beta_var_prior = 1, 
                      _gamma_prior = 1.5, 
                      _delta_prior = 0.5;
                      ndraw = 1500, 
                      burnin = 500, 
                      sigma_squared0 = Nothing, 
                      display = true, 
                      display_step = 250,
                      k = _k, beta_prior = _beta_prior, beta_var_prior = _beta_var_prior,
                      gamma_prior = _gamma_prior, delta_prior = _delta_prior)
                      
    T = size(y, 1)
    m = size(y, 2)

    # Create containers:
    sampled_beta = zeros(m, k, ndraw - burnin)
    sampled_sigma_squared = zeros(m, ndraw - burnin)
    sampled_factor = zeros(T, k, ndraw - burnin)

    # Initial values:
    sigma_squared = sigma_squared0 == Nothing ? ones(m) * (gamma_prior - 1) / delta_prior : sigma_squared0
    factor = factor_initialize(y, k)

    # Sampling:
        display ? println("Estimate normal linear factor model (via Gibbs sampling)") : -1
        for i = 1:ndraw
            (mod(i, display_step) == 0 && display) ? println(i) : -1

            # Sampling:
            beta = sampling_factor_loading(y, factor, beta_prior, beta_var_prior, Diagonal(sigma_squared))  # sampling factor loadings
            sigma_squared = sampling_sigma_squared(y - factor * beta', gamma_prior, delta_prior)  # sampling error variances
            factor = sampling_factor(y, beta, Diagonal(sigma_squared))  # sampling factors from normal distribution

            # Save samples:
            if i > burnin
                sampled_beta[:, :, i-burnin] = beta
                sampled_sigma_squared[:, i-burnin] = sigma_squared
                sampled_factor[:, :, i-burnin] = factor
            end
        end
        display ? println("Done") : -1
    return [sampled_beta, sampled_sigma_squared, sampled_factor]
end


"""
This component sets an initial values for the factors for the normal factor model by the following formula:

\$PCA_{1:k} Q R_{1:k}\$

where \$Q\$ and \$R\$ are the matrixes from QR decomposition and \$PCA\$ is the matrix of the principal components ordered by
the magnitude of the corresponding eigen values.
"""
function factor_initialize(x::AbstractMatrix, k::Int)
    pca_loading, pca_component = sorted_pca(mapslices(zscore,x,dims=1),2)
    q,r = qr(pca_loading')
    factor0 = pca_component * q * r[:,1:k]
    return factor0
end

"""
This component calculates the first \$k\$ principal components and the normalized loading matrix
"""
function sorted_pca(x::AbstractMatrix, k::Int)
    n = size(x,2)
    x_eigvecs = eigvecs(x'*x)
    pca_loading = sqrt(n) * x_eigvecs[:,n:-1:n-(k-1)]
    pca_component = x * pca_loading / n
    return [pca_loading, pca_component]
end
