module BayesianFactorModels

using LinearAlgebra

greet() = print("Hello World!")

include("filtering/kalman_filter.jl")
include("filtering/kalman_filter_tvp.jl")

end # module
