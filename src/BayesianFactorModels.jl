module BayesianFactorModels

using LinearAlgebra, MAT, LinearAlgebra, Distributions, Polynomials, SpecialFunctions

greet() = print("BayesianFactorModels Julia package")

export kalman_filter, kalman_filter_tvp, 
       dgp_normal, dgp_dynamic, 
       sampling_beta, sampling_carter_kohn, sampling_carter_kohn_tvp, sampling_df,
       sampling_factor, sampling_factor_dynamic, sampling_factor_loading,
       sampling_mixtrue_scale, sampling_sigma_squared,
       sampling_sv, sampling_sv_rw,
       dynamic_factor_model, dynamic_factor_model_sv,
       factor_model, linear_model, linear_model_t,
       sv_model, sv_model_rw

# Add DGP functions:
include("dgp/dgp_factor_model.jl")
include("dgp/dgp_dynamic_factor_model.jl")

# Add filtering functions:
include("filtering/kalman_filter.jl")
include("filtering/kalman_filter_tvp.jl")

# Add sampling functions:
include("sampling/sampling_beta.jl")
include("sampling/sampling_carter_kohn.jl")
include("sampling/sampling_carter_kohn_tvp.jl")
include("sampling/sampling_df.jl")
include("sampling/sampling_factor.jl")
include("sampling/sampling_factor_dynamic.jl")
include("sampling/sampling_factor_loading.jl")
include("sampling/sampling_factor_sv.jl")
include("sampling/sampling_mixture_scale.jl")
include("sampling/sampling_sigma_squared.jl")
include("sampling/sampling_sv.jl")
include("sampling/sampling_sv_rw.jl")

# Add models:
include("models/dynamic_factor_model.jl")
include("models/dynamic_factor_model_sv.jl")
include("models/factor_model.jl")
include("models/linear_model.jl")
include("models/linear_model_t.jl")
include("models/sv_model.jl")
include("models/sv_model_rw.jl")

end # module
