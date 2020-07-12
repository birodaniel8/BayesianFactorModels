"""
    sampling_mixtrue_scale(x, sigma_squared, v)

Sampling the mixture scale parameter of the normal linear model with independent normal-gamma priors and known
heteroscedasticity (ie t-errors)

It iterates trough all elements of the error matrix `x` and takes a sample from \$\\Gamma(\\alpha_v,\\beta_v)\$, where

\$\\alpha_v = \\frac{v + 1}{2}\$

\$\\beta_v = 2/\\frac{x^2}{\\sigma^2} + v\$

## Arguments
- `x::Vector::Matrix`: (T x m) error matrix of the regressions (conditioned on the coefficients)
- `sigma_squared::Number::Vector`: () or (m) static variance of the error terms (\$\\sigma^2\$)
- `v::Number::Vector`: () or (m) vector of degree of freedom parameters

## Returns
- `sampled_lambda::Vector`: (T x m) sampled mixture scale parameter
"""
function sampling_mixtrue_scale(x, 
                                _sigma_squared, 
                                _v;
                                sigma_squared = _sigma_squared, v = _v)
    t = size(x, 1)
    m = size(x, 2)

    # Sampling:
    sampled_lambda = zeros(t, m)
    for i = 1:m
        for j = 1:t
            alpha = (v[i] + 1) / 2
            beta = 2 / ((x[j, i] ^ 2) / sigma_squared[i] + v[i])
            sampled_lambda[j, i] = rand(Gamma(alpha, beta))
        end
    end
    return sampled_lambda
end
