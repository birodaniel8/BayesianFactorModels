"""
    kalman_filter_tvp(z, H, R, G, Q, mu=0, x0=0, P0=1)

This function performs the Kalman Filter estimation with time-varying parameters.

\$z_t = H_t x_t + \\epsilon_t \\quad \\quad \\epsilon_t \\sim N(0,R_t)\$
\$x_t = \\mu + G_t x_{t-1} + \\eta_t \\quad \\quad \\eta_t \\sim N(0,Q_t)\$

For detailed description see [`kalman_filter`](@ref).
"""
function kalman_filter_tvp(z, 
                           H, 
                           R, 
                           G, 
                           Q, 
                           _mu=0, 
                           _x0=0, 
                           _P0=1; 
                           mu = _mu, x0 = _x0, P0 = _P0)
                           
    z = z'
    T = size(z, 2)
    k = size(H, 2)
    x = zeros(k, T)
    P = zeros(k, k, T)
    x0 = isa(x0, Number) ? ones(k) * x0 : x0  # Number to vector
    P0 = isa(P0, Number) ? I(k) * P0 : P0  # Number to matrix

    # convert all parameter matrix into a 3 dim matrix:
    H = isa(H, Number) ? H = [H] : H
    R = isa(R, Number) ? R = [R] : R
    G = isa(G, Number) ? G = [G] : G
    Q = isa(Q, Number) ? Q = [Q] : Q
    H = size(H, 3) == 1 ? H = repeat(H, 1, 1, T) : H
    R = size(R, 3) == 1 ? R = repeat(R, 1, 1, T) : R
    G = size(G, 3) == 1 ? G = repeat(G, 1, 1, T) : G
    Q = size(Q, 3) == 1 ? Q = repeat(Q, 1, 1, T) : Q

    # Loop through and perform the Kalman filter equations recursively:
    for i = 1:T
        # Prediction step:
        if i == 1
            x[:, i] = mu .+ G[:,:,i] * x0  # Predict the state vector from the initial values
            P[:, :, i] = G[:,:,i] * P0 * G[:,:,i]' + Q[:,:,i]  # Predict the covariance from the initial values
        else
            x[:, i] = mu .+ G[:,:,i] * x[:, i-1]  # Predict the state vector
            P[:, :, i] = G[:,:,i] * P[:, :, i-1] * G[:,:,i]' + Q[:,:,i]  # Predict the covariance
        end
        # Update step:
        if all(isnan.(z[:, i]))
            # if all of them is missing at time 'i', there is not update and the updated values are the predicted ones
        elseif any(isnan.(z[:, i]))
            keep = ~isnan.(z[:, i])
            K = P[:, :, i] * H[keep, :, :]' / (H[keep, :, i] * P[:, :, i] * H[keep,: , i]' + R[keep,keep,i])  # Calculate the Kalman gain matrix only with the observed data
            x[:, i] = x[:, i] + K * (z[keep, i] - H[keep, :, i] * x[:, i])  # Update the state vector only with the observed data
            P[:, :, i] = (I(k) - K * H[keep, :, i]) * P[:, :, i]  # Update the covariance only with the observed data
        else
            K = P[:, :, i] * H[:,:,i]' / (H[:,:,i] * P[:, :, i] * H[:,:,i]' + R[:,:,i])  # Calculate the Kalman gain matrix
            x[:, i] = x[:, i] + K * (z[:, i] - H[:,:,i] * x[:, i])  # Update the state vector
            P[:, :, i] = (I(k) - K * H[:,:,i]) * P[:, :, i]  # Update the covariance
        end
    end
    x = x'
    return x, P
end
