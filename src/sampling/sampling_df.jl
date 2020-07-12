"""
    sampling_df(lambda, v_previous, v_prior = 30, hm_variance = 0.25)

Sampling the degree of freedom parameter of the normal linear model with independent normal-gamma priors and known
    heteroscedasticity (ie t-errors).

It iterates trough the columns of the matrix `lambda` and takes a sample from the distribution given as:

\$p(v|...) \\propto \\Big(\\frac{v}{2}\\Big)^{0.5Tv}\\Gamma\\Big(\\frac{v}{2}\\Big)^{-T}e^{-\\eta v}\$
where \$\\eta = \\frac{1}{v_{prior}} + 0.5\\sum_{i=1}^T[ln(\\lambda^{-1}) + \\lambda]\$ via random walk Metropolis-Hastings algorithm.


## Arguments
- `lambda::Vector::Matrix`: (T x m) mixture scale parameters (\$\\lambda\$)
- `v_previous::Number::Vector`: () or (m) vector of the previous degree of freedom parameters
- `v_prior::Number::Vector = 30`: () or (m) vector of prior degree of freedom parameters (\$v_{prior}\$)
- `hm_variance::Number::Vector = 0.25`: () or (m) random walk Metropolis-Hastings algorithm variance parameters

## Returns
- `v_sampled::Vector`: (m) sampled degree of freedoms
"""
function sampling_df(lambda, _v_previous, _v_prior = 30, _hm_variance = 0.25;
                     v_previous = _v_previous, v_prior = _v_prior, hm_variance = _hm_variance)
    t = size(lambda, 1)
    m = size(lambda, 2)

    # Transform inputs to the right format:
    v_prior = isa(v_prior, Number) ? ones(m) * v_prior : v_prior  # Number to vector
    hm_variance = isa(hm_variance, Number) ? ones(m) * hm_variance : hm_variance  # Number to vector

    # Sampling:
    v_sampled = zeros(m)
    for i = 1:m
        # Metropolis-Hastings sampling:
        v_proposed = v_previous[i] + sqrt(hm_variance[i]) * randn()
        eta = 1 / v_prior[i] + 0.5 * sum(-log.(lambda[:, i]) + lambda[:, i])
        # Calculate acceptance probability:
        if v_proposed > 0
            l_post_proposed = 0.5 * t * v_proposed * log(0.5 * v_proposed) - t * loggamma( 0.5 * v_proposed) - eta * v_proposed
            l_post_sampled = 0.5 * t * v_previous[i] * log.(0.5 * v_previous[i]) - t * loggamma.(0.5 * v_previous[i]) - eta * v_previous[i]
            alpha = exp.(l_post_proposed - l_post_sampled)
        else
            alpha = 0
        end

        if rand() < alpha
            v_sampled[i] = v_proposed
        else
            v_sampled[i] = v_previous[i]
        end
    end
    return v_sampled
end
