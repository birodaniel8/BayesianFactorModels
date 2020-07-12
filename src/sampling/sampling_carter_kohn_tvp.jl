"""
    sampling_carter_kohn_tvp(x, P, G, Q, mu=0, j=Nothing)

Carter & Kohn (1994) sampling algorithm for sampling Kalman Filtered states with time varying parameters.

For detailed description see [`sampling_carter_kohn`](@ref).
"""
function sampling_carter_kohn_tvp(x, P, G, Q, _mu=0, _j=Nothing; mu=_mu, j=_j)
    j = j == Nothing ? size(Q, 1) : j
    T = size(x,1)

    # convert all parameter matrix into a 3 dim matrix:
    isa(G, Number) ? G = [G] : G
    isa(Q, Number) ? Q = [Q] : Q
    G = size(G, 3) == 1 ? G = repeat(G, 1, 1, T) : G
    Q = size(Q, 3) == 1 ? Q = repeat(Q, 1, 1, T) : Q

    T = size(x,1)
    k = size(x,2)
    x = x'
    sampled_x = zeros(k,T)
    Q_star = Q[1:j, 1:j, :]
    G_star = G[1:j, :, :]
    for s = T:-1:1
        if s == T
            sampled_x[:, s] = rand(MultivariateNormal(x[:, s], Matrix(Hermitian(P[:,:,s]))))
        else
            x_star = x[:, s] + P[:, :, s] * G_star[:, :, s]' * inv(G_star[:, :, s]*P[:, :, s]*G_star[:, :, s]'+Q_star[:, :, s]) * (sampled_x[1:j, s+1] .- mu - G_star[:, :, s] * x[:, s])
            P_star = P[:, :, s] - P[:, :, s] * G_star[:, :, s]' * inv(G_star[:, :, s]*P[:, :, s]*G_star[:, :, s]'+Q_star[:, :, s]) * G_star[:, :, s] * P[:, :, s]
            sampled_x[:, s] = rand(MultivariateNormal(x_star, Matrix(Hermitian(P_star))))
        end
    end
    sampled_x = sampled_x'
    return sampled_x
end
