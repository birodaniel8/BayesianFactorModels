"""
    kalman_filter(z, H, R, G, Q, mu=0, x0=0, P0=1)

This function performs the Kalman Filter estimation.

\$z_t = H_t x_t + \\epsilon_t \\quad \\quad \\epsilon_t \\sim N(0,R_t)\$
\$x_t = \\mu + G_t x_{t-1} + \\eta_t \\quad \\quad \\eta_t \\sim N(0,Q_t)\$

## Arguments
- `z::Matrix`: (T x m) matrix of observations
- `H::Matrix`: (m x k x (T)) observation model matrix
- `R::Matrix`: (m x m x (T)) masurement noise covariance
- `G::Matrix`: (k x k x (T)) state transition matrix
- `Q::Matrix`: (k x k x (T)) process noise matrix
- `mu::Vector=0`: (k x 1) vector of constant terms
- `x0::Vector=0`: (k x 1) initial state values
- `P0::Matrix=1`: (k x k) initial state covariance matrix


## Returns
- `x::Matrix`: (T x k) estimated states
- `P::Matrix`: (k x k x T) estimated state covariance matrixes

Notes: If any of the input matrixes (H, R, G, Q) is more then 2 dimensional (ie the model has time-varying parameters),
then a [`kalman_filter_tvp`](@ref) is called.
This function can also handle missing values in the observation matrix.

The Kalman filter updates the state and the state covariance by a prediction-filtering procedure. The prediction propagates
`x` and `P` based on information available at `t-1`:

\$\\hat{x}_{t|t-1} = \\mu_t + G_t\\hat{x}_{t-1|t-1}\$
\$P_{t|t-1} = G_t P_{t-1|t-1} G_t' + Q_t\$

The updating step updates the estimates of `x` and `P` with information available at `t`:

\$K_t = P_t H_t'(H_t P_{t|t-1} H_t' + R_t)^{-1}\$
\$\\hat{x}_{t|t} = \\hat{x}_{t|t-1} + K_t(z_t-H_t\\hat{x}_{t|t-1})\$
\$P_{t|t} = (I - K_t H_t)P_{t|t-1}\$
"""
function kalman_filter(z, H, R, G, Q, _mu=0, _x0=0, _P0=1; mu = _mu, x0 = _x0, P0 = _P0)
    if size(H, 3) > 1 || size(R, 3) > 1 || size(G, 3) > 1 || size(Q, 3) > 1
        x, P = kalman_filter_tvp(z, H, R, G, Q, mu, x0, P0)
    else
        z = z'
        T = size(z, 2)
        k = size(H, 2)
        x = zeros(k, T)
        P = zeros(k, k, T)
        x0 = isa(x0, Number) ? ones(k) * x0 : x0  # Number to vector
        P0 = isa(P0, Number) ? I(k) * P0 : P0  # Number to matrix

        # Loop through and perform the Kalman filter equations recursively:
        for i = 1:T
            # Prediction step:
            if i == 1
                x[:, i] = mu .+ G * x0  # Predict the state vector from the initial values
                P[:, :, i] = G * P0 * G' + Q  # Predict the covariance from the initial values
            else
                x[:, i] = mu .+ G * x[:, i-1]  # Predict the state vector
                P[:, :, i] = G * P[:, :, i-1] * G' + Q  # Predict the covariance
            end
            # Update step:
            if all(isnan.(z[:, i]))
                # if all of them is missing at time 'i', there is not update and the updated values are the predicted ones
            elseif any(isnan.(z[:, i]))
                keep = .~isnan.(z[:, i])
                K = P[:, :, i] * H[keep, :]' / (H[keep, :] * P[:, :, i] * H[keep, :]' + R[keep,keep])  # Calculate the Kalman gain matrix only with the observed data
                x[:, i] = x[:, i] + K * (z[keep, i] - H[keep, :] * x[:, i])  # Update the state vector only with the observed data
                P[:, :, i] = (I(k) - K * H[keep, :]) * P[:, :, i]  # Update the covariance only with the observed data
            else
                K = P[:, :, i] * H' / (H * P[:, :, i] * H' + R)  # Calculate the Kalman gain matrix
                x[:, i] = x[:, i] + K * (z[:, i] - H * x[:, i])  # Update the state vector
                P[:, :, i] = (I(k) - K * H) * P[:, :, i]  # Update the covariance
            end
        end
        x = x'
    end
    return x, P
end
