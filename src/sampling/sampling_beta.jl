"""
    sampling_beta(y, x, beta_prior = 0, beta_var_prior = 1, sigma_squared = 0.01; last_truncated = false, stationarity_check = false, constant_included = true, max_iterations = 10000)

Sampling the \$\\beta\$ coefficients of the normal linear model with independent normal-gamma priors.

It takes a sample from \$N(\\beta_{posterior},V_{posterior})\$, where

\$V_{posterior} = (V_{prior}^{-1} + x'\\Sigma^{-1}x)^{-1}\$

\$\\beta_{posterior} = V_{posterior}(V_{prior}^{-1}\\beta_{prior} + x'\\Sigma^{-1}y)\$

## Arguments
- `y::Vector`: (T x 1) dependent variable
- `x::Matrix`: (T x k) explanatory variables
- `beta_prior::Number::Vector = 0`: () or (k) mean of the prior distribution (\$\\beta_{prior}\$)
- `beta_var_prior::Number::Vector::Matrix = 1`: () or (k) or (k x k) covariance of the prior distribution (\$V_{prior}\$)
- `sigma_squared::Number::Matrix = 0.01`: () or (1) or (T x T) error (co)variance (\$\\Sigma\$)
- `last_truncated::bool = false`: if true, the sample is taken from a multivariate normal distribution but the last element is truncated at 0
- `stationarity_check::bool = false`: if true, the sampling is repeated until the sampled coefficients stand for a stationary AR(k) model (without intercept)
- `constant_included::bool = false`: if true, the stationarity check is based on an AR(k-1) model (with intercept as a 1st variable)
- `max_iterations::Int = 10000`: maximum number of samples taken to sample coefficients for a stationary AR(k) model (without intercept)

## Returns
- `sampled_beta::Number::Vector`: () or (k x 1) sampled \$\\beta\$ coefficients
"""
function sampling_beta(y, x, _beta_prior = 0, _beta_var_prior = 1, _sigma_squared = 0.01;
                       last_truncated = false, stationarity_check = false, constant_included = false, max_iterations = 10000,
                       beta_prior = _beta_prior, beta_var_prior = _beta_var_prior, sigma_squared = _sigma_squared)
    k = size(x, 2)
    i0 = constant_included ? 2 : 1
    # Transform prior inputs to the right format:
    beta_prior = isa(beta_prior, Number) ? ones(k) * beta_prior : beta_prior  # Number to vector
    beta_var_prior = isa(beta_var_prior, Number) ? I(k) * beta_var_prior : beta_var_prior  # Number to array
    beta_var_prior = length(size(beta_var_prior)) == 1 ? Diagonal(beta_var_prior) : beta_var_prior  # Vector to array
    sigma_squared = length(size(sigma_squared)) == 1 ? Diagonal(sigma_squared) : sigma_squared  # Vector to array

    # Calculate posteriror parameters
    beta_var_posterior = inv(inv(beta_var_prior) .+ (x' * inv(sigma_squared) * x))
    beta_var_posterior = Matrix(Hermitian(beta_var_posterior))
    beta_posterior = beta_var_posterior * (inv(beta_var_prior) * beta_prior .+ x' * inv(sigma_squared) * y)

    # Sampling:
    sampled_beta = zeros(k)
    i = 1
    while true && i <= max_iterations
        if ~last_truncated
            # Sampling from multivariate normal:
            sampled_beta = rand(MultivariateNormal(beta_posterior, beta_var_posterior))
        else
            # sampling from multivariate normal with the last component truncated at 0:
            sampled_beta = [rand(MultivariateNormal(beta_posterior[1:k-1], beta_var_posterior[1:k-1,1:k-1]));
                            rand(truncated(Normal(beta_posterior[k],beta_var_posterior[k,k]),0,Inf))]
        end
        # check the stationarity of the estimated AR model if required:
        if (stationarity_check && all(abs.(roots(Polynomial([1;-sampled_beta[i0:k]]))).>1)) || ~stationarity_check
            break
        end
        i += 1
        if i > max_iterations
            error("The sampling procedure has reached the maximum number of iterations (no stationary solution sampled)")
        end
    end
    return sampled_beta
end
