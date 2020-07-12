"""
    dynamic_factor_model_sv()
"""
function dynamic_factor_model_sv(y, 
                                 _k, 
                                 _beta_prior = 0, 
                                 _beta_var_prior = 1,
                                 _rho_prior = 0, 
                                 _rho_var_prior = 10, 
                                 _gamma_tau_prior = 1.5, 
                                 _delta_tau_prior = 0.5,
                                 _theta_prior = 0, 
                                 _theta_var_prior = 1,
                                 _rho_factor_prior = 0,
                                 _rho_var_factor_prior = 10,
                                 _gamma_tau_factor_prior = 1.5,
                                 _delta_tau_factor_prior = 0.5;
                                 ndraw = 1500, 
                                 burnin = 500, 
                                 display = true, 
                                 display_step = 250,
                                 k = _k, beta_prior = _beta_prior, beta_var_prior = _beta_var_prior,
                                 rho_prior = _rho_prior, rho_var_prior = _rho_var_prior,
                                 gamma_tau_prior = _gamma_tau_prior, delta_tau_prior = _delta_tau_prior,
                                 theta_prior = _theta_prior, theta_var_prior = _theta_var_prior,
                                 rho_factor_prior = _rho_factor_prior, rho_var_factor_prior = _rho_var_factor_prior,
                                 gamma_tau_factor_prior = _gamma_tau_factor_prior,
                                 delta_tau_factor_prior = _delta_tau_factor_prior)
                                 
    T = size(y, 1)
    m = size(y, 2)

    # Create containers:
    sampled_beta = zeros(m, k, ndraw - burnin)
    sampled_rho = zeros(2, m, ndraw - burnin)
    sampled_tau = zeros(m, ndraw - burnin)
    sampled_h = zeros(T, m, ndraw - burnin)
    sampled_theta = zeros(k, ndraw - burnin)
    sampled_factor = zeros(T, k, ndraw - burnin)
    sampled_rho_f = zeros(2, k, ndraw - burnin)
    sampled_tau_f = zeros(k, ndraw - burnin)
    sampled_g = zeros(T, k, ndraw - burnin)

    # Initial values:
    theta = zeros(k)
    factor = factor_initialize(y, k) #
    rho = zeros(2, m)
    tau = ones(m) * delta_tau_prior/(gamma_tau_prior-1)
    h = zeros(T,m)
    theta = zeros(k)
    rho_f = zeros(2, k)
    tau_f = ones(k) * delta_tau_factor_prior/(gamma_tau_factor_prior-1)
    g = zeros(T,k)
    error_variance = zeros(m,m,T)
    factor_variance = zeros(k,k,T)

    # Sampling:
        display ? println("Estimate bayesian dynamic linear factor model (via Gibbs sampling)") : -1
        for i = 1:ndraw
            (mod(i, display_step) == 0 && display) ? println(i) : -1

            # Sampling factor loadings:
            error_variance[repeat(Matrix(I(m))[:],T)] = exp.(h)'[:]
            beta = sampling_factor_loading(y, factor, beta_prior, beta_var_prior, error_variance)

            # Sampling stochastic volatility components:
            e = y - factor * beta'
            for i = 1:m
                rho[:,i] = sampling_beta(h[2:T,i],[ones(T-1) h[1:T-1,i]],sigma_squared = tau[i],
                                         beta_prior = rho_prior, beta_var_prior = rho_var_prior,
                                         stationarity_check = true, constant_included=true)
                tau[i] = sampling_sigma_squared(h[2:T,i] .- rho[1,i] - rho[2,i] .* h[1:T-1,i], gamma_tau_prior, delta_tau_prior)[1]
                h[:,i] = sampling_sv(e[:,i], h[:,i], rho[:,i], tau[i], 0, 10)
            end

            # Sampling factor AR(1) coefficients:
            for j = 1:k
                factor_star = 1 ./(exp.(g[2:T,j]/2)) .* factor[2:T,j]
                factor_lag_star = 1 ./(exp.(g[2:T,j]/2)) .* factor[1:T-1,j]
                theta[j] = sampling_beta(factor_star,factor_lag_star,theta_prior,theta_var_prior,1,stationarity_check=true)[1]
            end

            # Sampling factors:
            error_variance[repeat(Matrix(I(m))[:],T)] = exp.(h)'[:]
            factor_variance[repeat(Matrix(I(k))[:],T)] = exp.(g)'[:]
            factor = sampling_factor_dynamic(y, beta, theta, error_variance, factor_variance)

            # Sampling factor stochastic volatility components:
            e_factor = factor[2:T,:] - factor[1:T-1,:]
            for j = 1:k
                rho_f[:,j] = sampling_beta(g[3:T,j],[ones(T-2) g[2:T-1,j]],sigma_squared = tau_f[j],
                                         beta_prior = rho_factor_prior, beta_var_prior = rho_var_factor_prior,
                                         stationarity_check = true, constant_included=true)
                tau_f[j] = sampling_sigma_squared(g[3:T,j] .- rho_f[1,j] - rho_f[2,j] .* g[2:T-1,j], gamma_tau_factor_prior, delta_tau_factor_prior)[1]
                g[2:T,j] = sampling_sv(e_factor[:,j], g[2:T,j], rho_f[:,j], tau_f[j], 0, 10)
                g[1,j] = g[2,j]
            end

            # Save samples:
            if i > burnin
                sampled_beta[:, :, i-burnin] = beta
                sampled_rho[:, :, i - burnin] = rho
                sampled_tau[:, i - burnin] = tau
                sampled_h[:, :, i - burnin] = h
                sampled_theta[:, i-burnin] = theta
                sampled_factor[:, :, i-burnin] = factor
                sampled_rho_f[:, :, i - burnin] = rho_f
                sampled_tau_f[:, i - burnin] = tau_f
                sampled_g[:, :, i - burnin] = g
            end
        end
        display ? println("Done") : -1
    return [sampled_beta, sampled_rho, sampled_tau, sampled_h, sampled_theta, sampled_factor, sampled_rho_f, sampled_tau_f, sampled_g]
end
