"""
    sampling_factor(y, beta, error_covariance)

Sampling the factors of the normal factor model with independent normal-gamma priors.

It samples the factors by taking independent samples at each time `t` from the distribution \$N(\\beta_f,V_f)\$, where

\$V_f = (I + \\beta'\\Sigma^{-1}\\beta)^{-1}\$

\$\\beta_f = V_f\\beta'\\Sigma^{-1}y)\$

## Arguments
- `y::Matrix`: (T x m) matrix of observations
- `beta::Vector::Matrix`: (m x k) estimated factor loadings (\$\\beta\$)
- `error_covariance::Matrix`: (m x m) error covariance matrix (\$\\Sigma\$)

## Returns
- `sampled_factor::Matrix`: (T x k) sampled factors

Note: this approach is equivalent with having independent \$N(0,1)\$ priors on each factor elements.
"""
function sampling_factor(y, 
                         beta, 
                         error_covariance)
                         
    T = size(y, 1)
    k = size(beta, 2)

    # Sampling factor values:
    sampled_factor = zeros(T, k)
    for t = 1:T
        factor_var = inv(I(k)+beta'*inv(error_covariance)*beta)
        factor_var = Matrix(Hermitian(factor_var))
        factor_mean = factor_var * beta' * inv(error_covariance) * y[t, :]
        factor_sampled_t = rand(MultivariateNormal(factor_mean, factor_var))
        sampled_factor[t, 1:k] = factor_sampled_t
    end
    return sampled_factor
end
