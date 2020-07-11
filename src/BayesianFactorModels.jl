module BayesianFactorModels

using LinearAlgebra, MAT

greet() = print("Hello World!")

export kalman_filter,kalman_filter_tvp

include("filtering/kalman_filter.jl")
include("filtering/kalman_filter_tvp.jl")

end # module
