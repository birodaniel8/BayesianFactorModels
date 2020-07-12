"""
    sampling_factor_loading(y, factor, beta_prior, beta_var_prior, error_covariance)

Sampling the \$\\beta\$ factor loadings of the normal factor model with independent normal-gamma priors.

It creates an `m` x `k` lower triangular matrix of factor loadings by taking sample row-by-row from a posterior distribution,
which is \$N(\\beta_{posterior,1:i},V_{posterior,1:i,1:i})1(\\beta_i>0)\$ if \$i\\le k\$, where

\$V_{posterior,1:i,1:i} = (V_{prior,1:i,1:i}^{-1} + f_{1:i}'\\Sigma_{1:i,1:i}^{-1}f_{1:i})^{-1}\$

\$\\beta_{posterior,1:i} = V_{posterior,1:i,1:i}(V_{prior,1:i,1:i}^{-1}\\beta_{prior,1:i} + f_{1:i}'\\Sigma_{1:i,1:i}^{-1}y)\$

and \$N(\\beta_{posterior},V_{posterior})\$ else, where

\$V_{posterior} = (V_{prior}^{-1} + f'\\Sigma^{-1}f)^{-1}\$

\$\\beta_{posterior} = V_{posterior}(V_{prior}^{-1}\\beta_{prior} + f'\\Sigma^{-1}y)\$

## Arguments
- `y::Matrix`: (T x m) matrix of observations
- `factor::Vector::Matrix`: (T x k) estimated factor values
- `beta_prior::Number::Vector::Matrix`: () or (m x k) mean of the prior distribution of factor loadings (\$\\beta_{prior}\$)
- `beta_var_prior::Number::Vector::Matrix`: () or (m x k) variance of the prior distributions (\$V_{prior}\$)
- `error_covariance::Matrix`: (m x m) error covariance matrix (\$\\Sigma\$)

## Returns
- `sampled_beta::Matrix`: (m x k) sampled \$\\beta\$ factor loadings

Note: all factor loadings are treated as independent random variables and we have independent \$N(\\beta_{i,j},V_{i,j})\$
priors for each of them.
"""
function sampling_factor_loading(y, 
                                 factor, 
                                 beta_prior, 
                                 beta_var_prior, 
                                 error_covariance)
    m = size(y, 2)
    k = size(factor, 2)
    # Transform prior inputs to the right format:
    beta_prior = isa(beta_prior, Number) ? ones(m, k) * beta_prior : beta_prior  # Number to array
    beta_var_prior = isa(beta_var_prior, Number) ? ones(m, k) * beta_var_prior : beta_var_prior  # Number to array

    # sampling
    sampled_beta = zeros(m, k)
    for i = 1:m
        if i <= k
            sampled_beta[i, 1:i] = sampling_beta(y[:, i], factor[:, 1:i], beta_prior[i, 1:i], Diagonal(beta_var_prior[i, 1:i]), error_covariance[i, i, :], last_truncated = true)
        else
            sampled_beta[i, :] = sampling_beta(y[:, i], factor, beta_prior[i, :], Diagonal(beta_var_prior[i, :]), error_covariance[i, i, :])
        end
    end
    return sampled_beta
end
