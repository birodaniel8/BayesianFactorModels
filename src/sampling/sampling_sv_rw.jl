"""
    sampling_sv_rw(e, sigma_squared, prev_h, tau_squared, h0=0, P0=1)

Sampling the stochastic volatility component of the random walk stochastic volatility model.

It utilizes the procedure of Kim et al (1998), which describes the stochastic volatility by a state space model written on
the log squared errors.
However the error term in the observation equation is \$log(\\chi_1)\$ distributed, which is approximated by a normal mixture
model. First we take the normal scale mixture parameter sample from a given discrete distribution.
This distribution is described in Kim et al (1998) and the parameters (\$q_{\\omega}^*, m_{\\omega}^*,{v_{\\omega}^*}^2)\$
are contained in Table 4 of the original paper.
Then we take a sample of the stochastic volatility via the Carter & Kohn (1994) algorithm from the following model:

\$e^*_t = h_t + u_t \\quad \\quad u_t \\sim N(0,{v_t^*}^2)\$

\$h_t = h_{t-1} + \\eta_t \\quad \\quad \\eta_t \\sim N(0,\\tau^2)\$

where \$e_t^* = log(e_t^2) - log(\\sigma^2) - m_{\\omega}^* + 1.2704\$.

## Arguments
- `e::Vector`: (T) error error vector
- `sigma_squared::Number`: static error variance
- `prev_h::Vector`: (T) stochastic variance component from the previous MCMC step
- `tau_squared::Number`: Variance of the stochastic volatility (\$\\tau^2\$)
- `h0:Number=0`: initial state value of the stochastic volatility
- `P0:Number=1`: initial state variance of the stochastic volatility

## Returns
- `sampled_h::Vector`: (T) sampled stochastic variance
"""
function sampling_sv_rw(e, sigma_squared, prev_h, tau_squared, h0=0, P0=1)
    table4 = [[1 0.00730 -10.12999 5.79596];
              [2 0.10556 -3.97281 2.61369];
              [3 0.00002 -8.56686 5.17950];
              [4 0.04395 2.77786 0.16735];
              [5 0.34001 0.61942 0.64009];
              [6 0.24566 1.79518 0.34023];
              [7 0.25750 -1.08819 1.26261]]
    c = 1.2704
    T = size(e, 1)
    any(e .== 0) ? println("The error vector contains a 0 value. Apply an offset to aviod it!") : -1

    # Sampling the mixture normal parameter:
    u_star = log.(e.^2) .- log(sigma_squared) - prev_h
    omega = zeros(T, 1)
    for t = 1:T
        q = map(x -> table4[x,2] * pdf(Normal(table4[x,3]-c, sqrt(table4[x,4])),u_star[t]),1:7)
        q = q ./ sum(q)
        q_prob = cumsum(q)
        omega[t] = sum(q_prob .< rand(1)) + 1
    end
    omega = convert.(Int,omega)

    # Sampling stochastic volatility:
    e_star = log.(e.^2) .- log(sigma_squared) - (table4[omega, 3] .- c)
    H = 1
    R = reshape(table4[omega, 4], (1,1,T))
    h_hat, P_hat = kalman_filter(e_star, H, R, 1, tau_squared, 0, h0, P0)
    sampled_h = sampling_carter_kohn(h_hat, P_hat, 1, tau_squared)
    return sampled_h
end
