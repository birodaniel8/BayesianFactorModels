"""
    sampling_sigma_squared(x, gamma_prior = 1.5, delta_prior = 0.5)

Sampling error variances of the normal linear model with independent normal-gamma priors.

It iterates trough the columns of the error matrix `x` and takes a sample from \$\\Gamma(\\gamma_{posterior},\\delta_{posterior})\$, where

\$\\gamma_{posterior} = \\frac{T}{2} + \\gamma_{prior}\$

\$\\delta_{posterior} = \\frac{x'x}{2} + \\delta_{prior}\$

## Arguments
- `x::Vector::Matrix`: (T x m) error matrix of the regressions (conditioned on the coefficients)
- `gamma_prior::Number::Vector = 1.5`: () or (m) shape of the prior distribution (\$\\gamma_{prior}\$)
- `delta_prior::Number::Vector = 0.5`: () or (m) scale of the prior distribution (\$\\delta_{prior}\$)

## Returns
- `sampled_sigma_squared::Vector`: (m x 1) sampled error variances

Note: the prior mean of the distribution is \$\\frac{\\gamma_{prior} - 1}{\\delta_{prior}}\$ (default = 1).
"""
function sampling_sigma_squared(x, _gamma_prior = 1.5, _delta_prior = 0.5;
                                gamma_prior = _gamma_prior, delta_prior = _delta_prior)
    t = size(x, 1)
    k = size(x, 2)

    # Calculate posterior parameters:
    gamma_posterior = ones(k) * t / 2 .+ gamma_prior
    if k == 1
        delta_posterior = (x' * x) / 2 .+ delta_prior
    else
        delta_posterior = diag(x' * x) / 2 .+ delta_prior
    end

    # Sampling:
    sampled_var = zeros(k)
    for i = 1:k
        sampled_var[i] = rand(InverseGamma(gamma_posterior[i], delta_posterior[i]))
    end
    return sampled_var
end
