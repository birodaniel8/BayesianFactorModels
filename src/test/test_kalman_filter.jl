include("C:\\Users\\birod\\Git\\BayesianFactorModels\\src\\BayesianFactorModels.jl")
using .BayesianFactorModels
using MAT
using LinearAlgebra
using Plots
# https://stackoverflow.com/questions/49662567/pykalman-default-handling-of-missing-values

## Import data:
y0 = matread("C:\\Users\\birod\\Documents\\Julia\\bayesian_factor_models\\kalman_filter_data.mat")["y0"][:,1];  # complete dataset
y = matread("C:\\Users\\birod\\Documents\\Julia\\bayesian_factor_models\\kalman_filter_data.mat")["y"][:,1];  # dataset with missing values
## Kalman filtering:
# transition_matrix
dt = 1;
F = [[1  dt   0.5 * dt * dt];
     [0   1          dt];
     [0   0           1]];
# observation_matrix
H = [1. 0. 0.];

# transition_covariance
Q = [[   1     0     0];
     [   0  1e-4     0];
     [   0     0  1e-6]];
     
# observation_covariance
R = [0.04];

# initial_state_mean
X0 = zeros(3);
# Kalman filter with non missing and missing values:

# initial_state_covariance
P0 = Diagonal([10.; 1.; 1.]);

x, P = kalman_filter(y0, H, R, F, Q, 0, X0, P0);
x2,P2 = kalman_filter(y,H,R,F,Q,0,X0,P0);

## Plot:
plot([x[:,1] + sqrt.(P[1,1,:]) x[:,1] x[:,1] - sqrt.(P[1,1,:])])
plot!([x2[:,1] + sqrt.(P2[1,1,:]) x2[:,1] x2[:,1] - sqrt.(P2[1,1,:])])