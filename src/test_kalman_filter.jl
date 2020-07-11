include("C:\\Users\\birod\\Git\\BayesianFactorModels\\src\\BayesianFactorModels.jl")
using .BayesianFactorModels
using MAT
using LinearAlgebra
# https://stackoverflow.com/questions/49662567/pykalman-default-handling-of-missing-values

# import data
y0 = matread("C:\\Users\\birod\\Documents\\Julia\\bayesian_factor_models\\kalman_filter_data.mat")["y0"][:,1];  # complete dataset
# y = matread("kalman_filter_data.mat")["y"][:,1]  # dataset with missing values

# transition_matrix
dt = 1;
F = [[1  dt   0.5*dt*dt];
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

x,P = kalman_filter(y0,H,R,F,Q,0,X0,P0);