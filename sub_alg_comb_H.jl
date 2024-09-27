using LoopVectorization
using Base: AbstractArrayOrBroadcasted
using OffsetArrays;
using LinearAlgebra;

include("sort_alg.jl")


# Poisson Sammpling
poisson(PI::AbstractArray{Float64}, pi_scale::Float64) = [rand() / pi_scale < pi for pi in PI]
poisson(PI::AbstractArray{Float64}) = [rand() < pi for pi in PI]
poisson_itr(PI, pi_scale::Float64) = Iterators.filter(x -> rand() / pi_scale < x, PI)
poisson_itr(PI) = Iterators.filter(x -> rand() < x, PI)
# # Defensive sampling with mixture rate α
function poisson(PI::AbstractArray{Float64}, pi_scale::Float64, α::Float64, r)
    a = 1/((1 - α) * pi_scale)
    b = - α/((1 - α) * pi_scale * r)
    return [(a * rand() + b) < pi for pi in PI]    
end


# Combining Samples
combine(M1::AbstractArray{Float64}, θ1::AbstractVector{Float64},
        M2::AbstractArray{Float64}, θ2::AbstractVector{Float64}) =
            (M1 + M2) \ (M1 * θ1 + M2 * θ2)

function ϕdd(y, d, θ, samp, PI::AbstractArray{Float64})
    dd = 0.0
    ϕ, σ = θ
    ϕ2 = abs2(ϕ)
    σ2 = abs2(σ)
    a = 1 - ϕ2
    
    @inbounds for j in samp
    # @turbo for i in eachindex(samp)
    #     j = samp[i]
        ej = y[j + 1] - ϕ^d[j] * y[j]
        b = 1 - ϕ2^d[j]

        dd += ((2 * d[j]^2 * ϕ^(2 * d[j] - 2) / b^2 - d[j] * ϕ^(2 * d[j] - 2) / b - (1 + ϕ^2) / (1 - ϕ^2)^2) +
            (ej^2 - 2 * ej * d[j] * y[j] * ϕ^d[j]) / σ^2 / b +
            ej^2 * ϕ^(2 * d[j]) * 2 * d[j] / σ^2 / b^2 +
            a * d[j]^2 * ϕ2^(d[j] - 1) * (2 * y[j + 1] * ej - y[j]^2 + y[j + 1]^2) / σ2 / b^2 +
            d[j] * ej * (y[j] - ϕ^d[j] * y[j + 1]) * ((d[j] - 1) * (((d[j] - 1) != 0) * ϕ^(d[j] - 2)) -
                                                    (d[j] + 1) * ϕ^d[j]) / σ2 / b^2 -
                                                    4 * a * d[j]^2 * ej^2 * ϕ2^(d[j] - 1) / σ2 / b^3)/PI[j]
    end
    # dd = O(m)
    return dd / length(d) * length(samp) # dd = O(r)
end

function ϕdd(y, d, θ, samp)
    dd = 0.0
    ϕ, σ = θ
    ϕ2 = abs2(ϕ)
    σ2 = abs2(σ)
    a = 1 - ϕ2
    
    @inbounds for j in samp
    # @turbo for i in eachindex(samp)
    #     j = samp[i]
        ej = y[j + 1] - ϕ^d[j] * y[j]
        b = 1 - ϕ2^d[j]

        dd += ((2 * d[j]^2 * ϕ^(2 * d[j] - 2) / b^2 - d[j] * ϕ^(2 * d[j] - 2) / b - (1 + ϕ^2) / (1 - ϕ^2)^2) +
            (ej^2 - 2 * ej * d[j] * y[j] * ϕ^d[j]) / σ^2 / b +
            ej^2 * ϕ^(2 * d[j]) * 2 * d[j] / σ^2 / b^2 +
            a * d[j]^2 * ϕ2^(d[j] - 1) * (2 * y[j + 1] * ej - y[j]^2 + y[j + 1]^2) / σ2 / b^2 +
            d[j] * ej * (y[j] - ϕ^d[j] * y[j + 1]) * ((d[j] - 1) * (((d[j] - 1) != 0) * ϕ^(d[j] - 2)) -
                                                    (d[j] + 1) * ϕ^d[j]) / σ2 / b^2 -
                                                    4 * a * d[j]^2 * ej^2 * ϕ2^(d[j] - 1) / σ2 / b^3)
    end
    # dd = O(r)
    return dd
end

function ϕdd(y, d, θ, bitsamp::BitArray)
    dd = 0.0
    ϕ, σ = θ
    ϕ2 = abs2(ϕ)
    σ2 = abs2(σ)
    a = 1 - ϕ2
    
    # @turbo for (j, bit) in enumerate(bitsamp)
    @inbounds for (j,bit) in enumerate(bitsamp)
        bit || continue
        ej = y[j + 1] - ϕ^d[j] * y[j]
        b = 1 - ϕ2^d[j]

        dd += bit * ((2 * d[j]^2 * ϕ^(2 * d[j] - 2) / b^2 - d[j] * ϕ^(2 * d[j] - 2) / b - (1 + ϕ^2) / (1 - ϕ^2)^2) +
            (ej^2 - 2 * ej * d[j] * y[j] * ϕ^d[j]) / σ^2 / b +
            ej^2 * ϕ^(2 * d[j]) * 2 * d[j] / σ^2 / b^2 +
            a * d[j]^2 * ϕ2^(d[j] - 1) * (2 * y[j + 1] * ej - y[j]^2 + y[j + 1]^2) / σ2 / b^2 +
            d[j] * ej * (y[j] - ϕ^d[j] * y[j + 1]) * ((d[j] - 1) * (((d[j] - 1) != 0) * ϕ^(d[j] - 2)) -
                                                    (d[j] + 1) * ϕ^d[j]) / σ2 / b^2 -
                                                    4 * a * d[j]^2 * ej^2 * ϕ2^(d[j] - 1) / σ2 / b^3)
    end

    return dd
end

function combine(dd1::Float64, θ1::AbstractVector{Float64}, lopt::Real,
                 dd2::Float64, θ2::AbstractVector{Float64}, lplt::Real)

    ϕ = (dd1 * θ1[1] + dd2 * θ2[1]) /(dd1 + dd2)
    -1 < ϕ < 1 || error("Out of bound")

    σ = sqrt((1-ϕ^2) *
        ((lopt * θ1[2]^2) / (1-θ1[1]^2)  + (lplt * θ2[2]^2)/ (1-θ2[1]^2))/ (lopt + lplt))
    return [ϕ, σ]
end


function suminv(x, ind)
    s = 0.0
    @turbo for i in eachindex(ind)
        s += 1 / x[ind[i]]
    end
    return s
end

function suminv(x, bitind::BitArray)
    s = 0.0
    @turbo for (i, bit) in enumerate(bitind)
        s += bit * 1 / x[i]
    end
    return s
end




 

## A-optmal Procedure
# Calculte Sampling Probabilities
function Cal_Mdd(yo, d, par, ind_itr)
    ϕ, σ = par

    # # Initiate variables
    # Mdd = zeros(Float64, 2, 2)
   
    # Speed up calculation
    ϕ2 = ϕ^2 
    σ2 = σ^2
    a = 1 - ϕ2
    
    n = 0
    dd1 = dd2 = dd3 = 0
    @inbounds for j in ind_itr
    # @turbo for i in eachindex(ind_itr)
    #     j = ind_itr[i]
        ej = yo[j+1] - ϕ^d[j] * yo[j]
        b = 1 - ϕ2^d[j]
        dd1 +=  (2 * d[j]^2 * ϕ^(2 * d[j] - 2) / b^2 - d[j] * ϕ^(2 * d[j] - 2) / b - (1 + ϕ^2) / (1 - ϕ^2)^2) +
            (ej^2 - 2 * ej * d[j] * yo[j] * ϕ^d[j]) / σ^2 / b +
            ej^2 * ϕ^(2 * d[j]) * 2 * d[j] / σ^2 / b^2 +
            a * d[j]^2 * ϕ2^(d[j] - 1) * (2 * yo[j+1] * ej - yo[j]^2 + yo[j+1]^2) / σ2 / b^2 +
            d[j] * ej * (yo[j] - ϕ^d[j] * yo[j+1]) * ((d[j] - 1) * (((d[j] - 1) != 0) * ϕ^(d[j] - 2)) -
            (d[j] + 1) * ϕ^d[j]) / σ2 / b^2 - 4 * a * d[j]^2 * ej^2 * ϕ2^(d[j] - 1) / σ2 / b^3

        dd2 += 2 * d[j] * ej * ϕ^(d[j] - 1) * (ϕ^d[j] * yo[j+1] - yo[j]) * (1 - ϕ2) / abs(σ)^3 / b^2 - 2 * ej^2 * ϕ / abs(σ)^3 / b
        dd3 += 1 / σ2 - 3 * ej^2 * (1 - ϕ2) / σ2^2 / b
        n += 1
    end
    # Mdd[2,1] = Mdd[1,2]
    Mdd = [dd1 dd2; dd2 dd3]
    # EVAL Mdd = O(r)
    return Mdd/ n # / length(ind_itr) # normalize to order O(1) to aovid numerical errors
end

function Cal_Mdd(yo, d, par, ind_itr, pw)
    ϕ, σ = par

    # Initiate variables
    Mdd = zeros(Float64, 2, 2)
   
    # Speed up calculation
    ϕ2 = ϕ^2 
    σ2 = σ^2
    a = 1 - ϕ2

    dd1 = dd2 = dd3 = 0
    @inbounds for j in ind_itr
    # @turbo for i in eachindex(ind_itr)
    #     j = ind_itr[i]
        ej = yo[j+1] - ϕ^d[j] * yo[j]
        b = 1 - ϕ2^d[j]
        
        dd1 += ((2 * d[j]^2 * ϕ^(2 * d[j] - 2) / b^2 - d[j] * ϕ^(2 * d[j] - 2) / b - (1 + ϕ^2) / (1 - ϕ^2)^2) +
            (ej^2 - 2 * ej * d[j] * yo[j] * ϕ^d[j]) / σ^2 / b +
            ej^2 * ϕ^(2 * d[j]) * 2 * d[j] / σ^2 / b^2 +
            a * d[j]^2 * ϕ2^(d[j] - 1) * (2 * yo[j+1] * ej - yo[j]^2 + yo[j+1]^2) / σ2 / b^2 +
            d[j] * ej * (yo[j] - ϕ^d[j] * yo[j+1]) * ((d[j] - 1) * (((d[j] - 1) != 0) * ϕ^(d[j] - 2)) -
            (d[j] + 1) * ϕ^d[j]) / σ2 / b^2 - 4 * a * d[j]^2 * ej^2 * ϕ2^(d[j] - 1) / σ2 / b^3)/ pw[j]

        dd2 += (2 * d[j] * ej * ϕ^(d[j] - 1) * (ϕ^d[j] * yo[j+1] - yo[j]) * (1 - ϕ2) / abs(σ)^3 / b^2 - 2 * ej^2 * ϕ / abs(σ)^3 / b)/pw[j]
        dd3 += (1 / σ2 - 3 * ej^2 * (1 - ϕ2) / σ2^2 / b)/pw[j]
    end
    Mdd = [dd1 dd2; dd2 dd3]
    return Mdd ./ length(d) # normalize to order O(1) to aovid numerical errors
end


function norm_z(yo, d, par, Mdd::AbstractArray{Float64})
    z = Array{Float64}(undef, length(d))

    ϕ, σ = par
    ϕ2, σ2 = par.^2
    den = (Mdd[1,1] * Mdd[2,2] - Mdd[1,2]^2)^2
    nu1 = (Mdd[1,2]^2 + Mdd[2,2]^2) / den
    nu2 = (2 * Mdd[1,2] * (Mdd[1,1]  + Mdd[2,2])) / den
    nu3 = (Mdd[1,1]^2 + Mdd[1,2]^2) / den

    ld = Array{Float64}(undef, 2)  
    @inbounds for j in Base.OneTo(length(z))        
        ej = yo[j+1] - ϕ^(d[j]) * yo[j]

        ζj = σ2 * (1 - ϕ^d[j]) / (1 - ϕ2)

        ld[1] = ((ej^2 - ζj) / (1 - ϕ2) / ζj * (ϕ^2 * ζj - d[j]σ2) + (yo[j+1] * ej - ζj)d[j]) / ϕ / ζj
        ld[2] = (ej^2/ ζj - 1) / σ
        z[j] = sqrt(nu1 * ld[1]^2 - nu2 * ld[1] * ld[2] + nu3 * ld[2]^2)
        # @inbounds z[j] = norm(Mdd \ ld)
    end

    return z
end


# Obtain estimates
function Aopt(y::AbstractVector, d::AbstractVector, r::Int, θplt::AbstractVector, Mdd_plt::AbstractMatrix, rplt::Int;
              Tlr= 1e-7, MIte::Int = 100, κ_fun::Function = find_κ, st_pt = [0.0, 1.0])
    z = norm_z(y, d, θplt, Mdd_plt)

    z_scale = κ_fun(z, r)

    samp = poisson(z, z_scale)

    PI = min.(z, 1/z_scale) # PI * z_scale = pi

    θopt, ite = IPW(y, d, st_pt, samp, PI; Tlr = Tlr, MIte = MIte)

    
    Mdd_opt = Cal_Mdd(y, d, θopt, findall(samp), min.(z .* z_scale, 1))
    
    return combine(rplt .* Mdd_plt, θplt, r .* Mdd_opt, θopt), ite   
end

function Aopt(y::AbstractVector, d::AbstractVector, r::Int, plt_itr;
              st_pt = [0.5, 1.0],  Tlr= 1e-7, MIte = 100, κ_fun::Function = find_κ)
    θplt, = MLE(y, d, st_pt, plt_itr)
    
    Mdd_plt = Cal_Mdd(y, d, θplt, plt_itr)
    
    z = norm_z(y, d, θplt, Mdd_plt)

    z_scale = κ_fun(z, r)

    samp = poisson(z, z_scale)

    PI = min.(z, 1/z_scale) # PI * z_scale = pi
    
    θopt, ite = IPW(y, d, st_pt, samp, PI; Tlr = Tlr, MIte = MIte)

    Mdd_opt = Cal_Mdd(y, d, θopt, findall(samp), min.(z .* z_scale, 1))
    
    return combine(length(plt_itr) .* Mdd_plt, θplt, r .* Mdd_opt, θopt), ite
end

function Aopt(y::AbstractVector, d::AbstractVector, r0::Int, r::Int;
              st_pt = [0.5, 1.0], Tlr= 1e-7, MIte = 100, κ_fun::Function = find_κ)
    plt_itr = # findall(poisson(repeat([r0/ length(d)], length(d)), 1.0))
        sample(1 : length(d), r0; replace = false)       
    return Aopt(y, d, r, plt_itr; st_pt = st_pt, Tlr= Tlr, MIte = MIte, κ_fun = κ_fun)
end

# Obtain A-opt Estimates under Mixture α

function Aopt(y::AbstractVector, d::AbstractVector, r::Int, θplt::AbstractVector, Mdd_plt::AbstractMatrix, rplt::Int,α::Float64;
              Tlr= 1e-7, MIte::Int = 100, κ_fun::Function = find_κ, st_pt = [0.0, 1.0])
    z = norm_z(y, d, θplt, Mdd_plt)

    z_scale = κ_fun(z, r)

    n = length(d)

    PI = min.(z .* z_scale .* (1 - α) .+ (α * r / n), 1)

    samp = poisson(PI)
    
    θopt, ite = IPW(y, d, st_pt, samp, PI; Tlr = Tlr, MIte = MIte)

    Mdd_opt = Cal_Mdd(y, d, θopt, findall(samp), PI)
    
    return combine(rplt .* Mdd_plt, θplt, r .* Mdd_opt, θopt), ite
end

function Aopt(y::AbstractVector, d::AbstractVector, r::Int, plt_itr, α::Float64;
              st_pt = [0.5, 1.0],  Tlr= 1e-7, MIte = 100, κ_fun::Function = find_κ)
    θplt, = MLE(y, d, st_pt, plt_itr)
    
    Mdd_plt = Cal_Mdd(y, d, θplt, plt_itr)
    
    z = norm_z(y, d, θplt, Mdd_plt)
    
    z_scale = κ_fun(z, r)

    n = length(d)

    PI = min.(z .* z_scale .* (1 - α) .+ (α * r / n), 1)

    samp = poisson(PI)
    
    θopt, ite = IPW(y, d, st_pt, samp, PI; Tlr = Tlr, MIte = MIte)

    Mdd_opt = Cal_Mdd(y, d, θopt, findall(samp), PI)

    return combine(length(plt_itr) .* Mdd_plt, θplt, r .* Mdd_opt, θopt), ite
end

function Aopt(y::AbstractVector, d::AbstractVector, r0::Int, r::Int, α::Float64;
              st_pt = [0.5, 1.0], Tlr= 1e-7, MIte = 100, κ_fun::Function = find_κ)

    plt_itr = sample(1 : length(d), r0; replace = false)       
    return Aopt(y, d, r, plt_itr, α; st_pt = st_pt, Tlr= Tlr, MIte = MIte, κ_fun = κ_fun)
end



# IBOSS-ISAR

function iboss_δ(yo, d, par, r)
    m = length(d)
    detI = Array{Float64}(undef, m)
    
    ϕ, σ = par
   
    # Speed up calculation
    ϕ2 = ϕ^2
    σ2 = σ^2
    o_m_ϕ2 = 1 - ϕ2

    ## CAREFUL ABOUT THE INDEX OF y[j - 1]
    Ij = Array{Float64}(undef, 2,2)
    # Reparameterization

    @inbounds for j in 1:m
        dj = d[j]
        yjm = yo[j]
        
        ςj = σ2 * (1 - ϕ2^dj) / o_m_ϕ2
        
        J12_ast = 2 / o_m_ϕ2 * (ϕ - σ2 * dj * ϕ^(2 * dj - 1) / ςj)
        Ij[1, 1] = (dj * yjm * ϕ^(dj - 1))^2 / ςj + J12_ast^2 / 2
        Ij[1, 2] = Ij[2, 1] = J12_ast / σ
        Ij[2, 2] = 2 / σ2
        # detI[j] = det(Ij)
        detI[j] = Ij[1,1] * Ij[2,2] - Ij[1,2]^2# det(Ij)
    end

    detI_thres = opt_select(detI, m - r)
    
    return detI .> detI_thres # / length(ind_itr) # normalize to order O(1) to aovid numerical errors
end

function iboss(y::AbstractVector, d::AbstractVector, r::Int, plt_itr;
               st_pt = [0.5, 1.0],  Tlr= 1e-7, MIte = 100)
    θplt, = MLE(y, d, st_pt, plt_itr)
    
    δ = iboss_δ(y, d, θplt, r)

    
    θiboss, ite = MLE(y, d, st_pt, δ; Tlr = Tlr, MIte = MIte)

    Mdd_plt = Cal_Mdd(y, d, θiboss, plt_itr)
    Mdd_iboss = Cal_Mdd(y, d, θiboss, findall(δ))
    
    return combine(length(plt_itr) .* Mdd_plt, θplt, r .* Mdd_iboss, θiboss), ite
end

function iboss(y::AbstractVector, d::AbstractVector, r0::Int, r::Int;
               st_pt = [0.5, 1.0],  Tlr= 1e-7, MIte = 100)
    plt_itr = sample(1 : length(d), r0; replace = false)        
    return iboss(y::AbstractVector, d::AbstractVector, r::Int, plt_itr; st_pt = st_pt,  Tlr = Tlr, MIte = MIte)
end

# Thin
function Cal_ISAR_FI!(FI::Array{Float64}, yjm::Float64, dj, par_cal::AbstractVector{Float64})
    # Speed up calculation
    ϕ, σ, ϕ2, σ2, o_m_ϕ2 = par_cal
    ςj = σ2 * (1 - ϕ2^dj) / o_m_ϕ2

    J12_ast = 2 / o_m_ϕ2 * (ϕ - σ2 * dj * ϕ^(2 * dj - 1)/ ςj)
    FI[1,1] = (dj * yjm * ϕ^(dj - 1))^2 / ςj + J12_ast^2 / 2
    FI[1,2] = FI[2,1] = J12_ast / σ
    FI[2,2] = 2 / σ2
    nothing
end

function Cal_ISAR_FI!(FI::Array{Float64}, yo::AbstractArray{Float64}, d::AbstractArray, j, par_cal::AbstractVector{Float64})
    # Speed up calculation
    ϕ, σ, ϕ2, σ2, o_m_ϕ2 = par_cal
    @inbounds ςj = σ2 * (1 - ϕ2^d[j]) / o_m_ϕ2
    @inbounds J12_ast = 2 / o_m_ϕ2 * (ϕ - σ2 * d[j] * ϕ^(2 * d[j] - 1)/ ςj)
    @inbounds FI[1,1] = (d[j] * yo[j] * ϕ^(d[j] - 1))^2 / ςj + J12_ast^2 / 2
    @inbounds FI[1,2] = FI[2,1] = J12_ast / σ
    @inbounds FI[2,2] = 2 / σ2
    nothing
end

trans(θ) = [θ[1], θ[2], θ[1]^2, θ[2]^2, 1 - θ[1]^2]

function Sum_ISAR_FI(yo, d, par_cal, ind_itr)
    ϕ, σ, ϕ2, σ2, o_m_ϕ2 = par_cal    
    FI_sum = zeros(2, 2)

    @inbounds for j in ind_itr
        ςj = σ2 * (1 - ϕ2^d[j]) / o_m_ϕ2
        J12_ast = 2 / o_m_ϕ2 * (ϕ - σ2 * d[j] * ϕ^(2 * d[j] - 1) / ςj)
        FI_sum[1, 1] = FI_sum[1, 1] + (d[j] * yo[j] * ϕ^(d[j] - 1))^2 / ςj + J12_ast^2 / 2
        FI_sum[2, 1] = FI_sum[2, 1] + J12_ast / σ
        FI_sum[2, 2] = FI_sum[2, 2] + 2 / σ2
        # Cal_ISAR_FI_add!(FI_sum, yo[j], d[j], par_cal)
    end
    
    FI_sum[1, 2] = FI_sum[2, 1]
    
    return FI_sum
end

function thin_δ(yo::AbstractVector, d::AbstractVector, r0::Int, r::Int, q::Float64, γ::Float64, ϵ::Float64; st_pt = LS_ISAR_st)
    # Calculate Parameters
    m = length(d)
    α = r / (m - r0)
    # Parameter value check
    ((0.5 < q < 1) && (0 < γ < q - 0.5) && (0 < ϵ < α)) || error("Violation of parameter requirements.")

    # Indicators
    δ = BitArray(undef, m)

    # Average Fisher information in the subdata
    par_plt, = MLE(yo, d, st_pt, 1:r0)
    par_cal = Array{Float64}(undef, 5)
    par_cal[1] = par_plt[1]
    par_cal[2] = par_plt[2]
    par_cal[3] = par_plt[1]^2
    par_cal[4] = par_plt[2]^2
    par_cal[5] = 1 - par_cal[3]
    
    FI_sub = Sum_ISAR_FI(yo, d, par_cal, 1:r0) ./ r0

    # Initialization
    ζ = Array{Float64}(undef, r0)
    Ik = Array{Float64}(undef, 2, 2)
    for k in 1 : r0
        Cal_ISAR_FI!(Ik, yo[k], d[k], par_cal)
        ζ[k] = (FI_sub[1,1] * Ik[2,2] + FI_sub[2,2] * Ik[1,1] - 2 * FI_sub[1,2] * Ik[1,2]) /
            (FI_sub[1,1] * FI_sub[2,2] - FI_sub[1,2]^2)# tr(FI_sub \ Ik) - 2
        δ[k] = 0
    end
    
    r0p = ceil(Int, (1 - α/2) * r0)
    r0m = max(floor(Int, (1 - 3 * α / 2) * r0), 1)
    β0 = r0 / (r0p - r0m)
    sort!(ζ)
    hl = opt_select(ζ, r0m)
    tmp = filter(>(hl), ζ)
    r0c = ceil(Int, (1 - α) * r0)
    Ck = r0c != r0m ? opt_select(tmp, r0c - r0m) : hl
    filter!(>(Ck), tmp)
    h = opt_select!(tmp, r0p - r0c) - hl
    

    rk = 0
    fk = let hk = h/r0^γ
        sum(abs.(ζ .- Ck) .< hk) / (2 * r0 * hk) end;

    @inbounds for k in (r0 + 1):m
        Cal_ISAR_FI!(Ik, yo[k], d[k], par_cal)
        ζk = (FI_sub[1,1] * Ik[2,2] + FI_sub[2,2] * Ik[1,1] - 2 * FI_sub[1,2] * Ik[1,2]) /
            (FI_sub[1,1] * FI_sub[2,2] - FI_sub[1,2]^2)# tr(FI_sub \ Ik) - 2
        δ[k] = (rk <= max(k * ϵ - r0, r - m + k - 1)) || ((rk < r) && (ζk >= Ck)) # no need k - 1 there for ϵ
        
        
        rk += δ[k] # CAREFUL ABOUT THE ORDER OF THIS LINE
        Ck, fk = let hk = h/k^γ
            Ck + min(1/fk, β0 * (k-1)^γ) * ((ζk >= Ck) - α) / k^q, fk + ((abs(ζk - Ck) < hk) / (2 * hk) - fk) * k^(-q)
        end
        upd = δ[k] / (rk  + r0)
        FI_sub[1,1] = FI_sub[1,1] + upd * (Ik[1,1] - FI_sub[1,1]) # USING rk since rk was updated
        FI_sub[1,2] = FI_sub[1,2] + upd * (Ik[1,2] - FI_sub[1,2]) # USING rk since rk was updated
        FI_sub[2,2] = FI_sub[2,2] + upd * (Ik[2,2] - FI_sub[2,2]) # USING rk since rk was updated
        
    end
    
    return δ, par_plt
end

function thin(y::AbstractVector, d::AbstractVector,r0::Int, r::Int, q, γ, ϵ; st_pt = [0.5, 1.0],  Tlr= 1e-7, MIte = 100)
    # Use the first r0 samples for pilot estimate
    δ, θplt = thin_δ(y, d, r0, r, q, γ, ϵ)

    θthin, ite = MLE(y, d, st_pt, δ; Tlr = Tlr, MIte = MIte)

    Mdd_plt = Cal_Mdd(y, d, θthin, 1:r0)
    Mdd_thin = Cal_Mdd(y, d, θthin, findall(δ))
    return combine(r0 .* Mdd_plt, θplt, r .* Mdd_thin, θthin), ite

end
