"""
    sampling_factor_dynamic(y, beta, theta, error_covariance, factor_covariance=1)

Sampling the factors of the bayesian dynamic normal factor model. It first gets the Kalman Filter state estimates, then a
sample is taken using the Carter & Kohn (1994) sampling algorithm.

\$y_t = f_t \\beta_t' + \\epsilon_t \\quad \\quad \\epsilon_{t,i} \\sim N(0,\\sigma^2_i)\$

\$f_{t,j} = \\theta_j f_{t,j-1} +\\eta_{t,j} \\quad \\quad \\eta_{t,j} \\sim N(0,1)\$

where \$i = 1...m\$, \$j = 1...k\$.

## Arguments
- `y::Matrix`: (T x m) matrix of observations
- `beta::Matrix`: (m x k x (T)) factor loadings
- `theta::Vector`: (k) autoregressive coefficient of the factors
- `error_covariance::Matrix`: (m x m x (T)) covariance matrix of the observation equation
- `factor_covariance::Matrix`: (k x k x (T)) covariance matrix of the factor equation

## Returns
- `sampled_factor::Matrix`: (T x k) sampled factor values
"""
function sampling_factor_dynamic(y, 
                                 beta, 
                                 theta, 
                                 error_covariance, 
                                 factor_covariance=1)
    k = size(beta, 2)

    # Set parameters:
    factor_covariance = isa(factor_covariance, Number) ? I(k) * factor_covariance : factor_covariance  # Number to matrix
    theta = isa(theta, Number) ? [theta] : theta  # Number to matrix

    H = beta
    R = error_covariance
    G = Diagonal(theta)
    Q = factor_covariance
    x0 = zeros(k)
    P0 = I(k)

    # Sampling:
    F, P = kalman_filter(y, H, R, G, Q, 0, x0, P0)
    sampled_factor = sampling_carter_kohn(F, P, G, Q)
    return sampled_factor
end
