"""
    sampling_carter_kohn(x, P, G, Q, mu=0, j=Nothing)

Carter & Kohn (1994) sampling algorithm for sampling Kalman Filtered states.

The Carter & Kohn procedure is a backward sampling algorithm where we recursively take \$x_t^*\$ samples from
\$N(\\bar{x}_{t|t}, \\bar{P}_{t|t})\$, where \$\\bar{x}_{T|T} = x_{T|T}\$ and \$\\bar{P}_{T|T} = P_{T|T}\$ if \$t = T\$ and

\$\\bar{x}_{t|t} = x_{t|t} + P_{t|t}\\hat{G}_t'[\\hat{G}_t P_{t|t} \\hat{G}_t' + \\hat{Q}_t]^{-1}[x_{t+1}^* - \\mu - \\hat{G}_t' x_{t|t}]\$
\$\\bar{P}_{t|t} = P_{t|t} - P_{t|t}\\hat{G}_t'[\\hat{G}_t P_{t|t} \\hat{G}_t' + \\hat{Q}_t]^{-1} \\hat{G}_t' P_{t|t}]\$

where
\$\\hat{G}_t' = G_{1:j,:}\$ and \$\\hat{Q}_t' = Q_{1:j,1:j}\$, where \$j\$ is the largest integer for which Q is positive definite.

## Arguments
- `x::Matrix`: (T x k) state estimations from Kalman Filter (updated values)
- `P::Matrix`: (k x k x T) state covariance from Kalman Filter (updated values)
- `G::Matrix`: (k x k x (T)) state transition matrix
- `Q::Matrix`: (k x k x (T)) process noise matrix
- `mu::Vector=0`: (k x 1) vector of constant terms
- `j::Int=Nothing`: size of the block for which the Q matrix is positive definite default: size(Q,1) ie. whole matrix

## Returns
- `sampled_x::Matrix`: (T x k) sampled states
"""
function sampling_carter_kohn(x, 
                              P, 
                              G, 
                              Q, 
                              _mu=0, 
                              _j=Nothing; 
                              mu=_mu, j=_j)

    j = j == Nothing ? size(Q, 1) : j

    if size(G, 3) > 1 || size(Q, 3) > 1
        sampled_x = sampling_carter_kohn_tvp(x, P, G, Q, mu, j)
    else
        T = size(x,1)
        k = size(x,2)
        x = x'
        sampled_x = zeros(k,T)
        isa(Q, Number) ? Q = [Q] : Q
        isa(G, Number) ? G = [G] : G
        Q_star = Q[1:j, 1:j]
        G_star = G[1:j, :]
        for s = T:-1:1
            if s == T
                sampled_x[:, s] = rand(MultivariateNormal(x[:, s], Matrix(Hermitian(P[:,:,s]))))
            else
                x_star = x[:, s] + P[:, :, s] * G_star' * inv(G_star*P[:, :, s]*G_star'+Q_star) * (sampled_x[1:j, s+1] .- mu - G_star * x[:, s])
                P_star = P[:, :, s] - P[:, :, s] * G_star' * inv(G_star*P[:, :, s]*G_star'+Q_star) * G_star * P[:, :, s]
                sampled_x[:, s] = rand(MultivariateNormal(x_star, Matrix(Hermitian(P_star))))
            end
        end
        sampled_x = sampled_x'
    end
    return sampled_x
end
