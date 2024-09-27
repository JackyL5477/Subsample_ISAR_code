using Base: test_success
using StatsBase
using LinearAlgebra;
using Statistics;
using OffsetArrays

## Newton-Ralphson Method
function  NR(
    d::Function, dd::Function, St_pt::AbstractVector{Float64};
    Tlr::Float64 = 1e-8, MIte::Int = 300)

    # St_pt is the starting point of Newton's method
    # Tlr is the tolerance for accepting convergence
    # MIte is the maximum number of iterations

    # Initiate
    opm = copy(St_pt); # starting point
    for ite in 1:MIte
        # Difference by Newton's
        opm_diff = dd(opm)\ d(opm)
        opm .-= opm_diff

        # # Check the convergence
        if mean(abs2, opm_diff) < Tlr
            return opm, ite
            break
        end
        
    end

    error("Failed convergence")
    return opm, MIte
end

# Taking single function for both first and second degree gradient
function NR(
    D::Function, St_pt::AbstractVector{Float64};
    Tlr::Float64 = 1e-8, MIte::Int = 300)
    # Initiate
    opm = copy(St_pt); # starting point
    # converge = true;
    for ite in 1:MIte
        # Difference by Newton's
        opm_diff = let D = D(opm)
            D[2] \ D[1] 
        end
        
        opm .-= opm_diff

        if mean(abs2, opm_diff) < Tlr
            return opm, ite
            break
        end
        
    end

    error("Failed convergence")
    return opm, MIte
end

function NR(
    D::Function, St_pt::Float64;
    Tlr::Float64 = 1e-8, MIte::Int = 300)
    # Initiate
    opm = copy(St_pt); # starting point
    for ite in 1:MIte
        # Difference by Newton's
        opm_diff = let D = D(opm)
            D[1] / D[2]
        end
        
        opm -= opm_diff

        if mean(abs2, opm_diff) < Tlr 
            return opm, ite
            break
        end
        
    end

    error("Failed convergence")
    return opm, MIte
end

function z_upd(z, σ, y, d, itr; wf::Function = fid)
    ϕ = tanh(z)
    # Initiate variables
    dd = 0.0
    fd = 0.0
    
    # Speed up calculation
    ϕ2 = abs2(ϕ)
    σ2 = abs2(σ)
    a = 1 - ϕ2

    ## CAREFUL ABOUT THE INDEX OF y[j - 1]
    for j in itr
        ej = y[j + 1] - ϕ^d[j] * y[j]
        b = 1 - ϕ2^d[j]

        fd += wf(d[j] * ϕ^(2 * d[j] - 1) / b - ϕ / a + ϕ * ej^2 / σ2 / b + 
            a * d[j] * ej * (ϕ^(d[j] - 1) * y[j] - ϕ^(2 * d[j] - 1) * y[j + 1]) / σ2 / b^2, j)
        
        dd += wf((2 * d[j]^2 * ϕ2^(d[j] - 1) / b^2 - d[j] * ϕ2^(d[j] - 1) / b - (1 + ϕ^2) / (1 - ϕ2)^2) +
            (ej^2 - 2 * ej * d[j] * y[j] * ϕ^d[j]) / σ2 / b +
            ej^2 * ϕ2^d[j] * 2 * d[j] / σ2 / b^2 +
            a * d[j]^2 * ϕ2^(d[j] - 1) * (2 * y[j + 1] * ej - y[j]^2 + y[j + 1]^2) / σ2 / b^2 +
            d[j] * ej * (y[j] - ϕ^d[j] * y[j + 1]) * ((d[j] - 1) * (((d[j] - 1) != 0) * ϕ^(d[j] - 2)) -
                                                    (d[j] + 1) * ϕ^d[j]) / σ2 / b^2 -
                                                    4 * a * d[j]^2 * ej^2 * ϕ2^(d[j] - 1) / σ2 / b^3, j)
    end

    return [1, ((1 - ϕ2) * dd / fd - 2 * ϕ)]
end

function ϕ_upd(ϕ, σ, y, d, itr; wf::Function = fid)
    -1 < ϕ < 1 || error("Out of bound")
    # Initiate variables
    dd = 0.0
    fd = 0.0

    
    # Speed up calculation
    ϕ2 = abs2(ϕ)
    σ2 = abs2(σ)
    a = 1 - ϕ2

    ## CAREFUL ABOUT THE INDEX OF y[j - 1]
    for j in itr
        ej = y[j + 1] - ϕ^d[j] * y[j]
        b = 1 - ϕ2^d[j]

        fd += wf(d[j] * ϕ^(2 * d[j] - 1) / b - ϕ / a + ϕ * ej^2 / σ2 / b + 
            a * d[j] * ej * (ϕ^(d[j] - 1) * y[j] - ϕ^(2 * d[j] - 1) * y[j + 1]) / σ2 / b^2, j)
        
        dd += wf((2 * d[j]^2 * ϕ^(2 * d[j] - 2) / b^2 - d[j] * ϕ^(2 * d[j] - 2) / b - (1 + ϕ^2) / (1 - ϕ^2)^2) +
            (ej^2 - 2 * ej * d[j] * y[j] * ϕ^d[j]) / σ^2 / b +
            ej^2 * ϕ^(2 * d[j]) * 2 * d[j] / σ^2 / b^2 +
            a * d[j]^2 * ϕ2^(d[j] - 1) * (2 * y[j + 1] * ej - y[j]^2 + y[j + 1]^2) / σ2 / b^2 +
            d[j] * ej * (y[j] - ϕ^d[j] * y[j + 1]) * ((d[j] - 1) * (((d[j] - 1) != 0) * ϕ^(d[j] - 2)) -
                                                    (d[j] + 1) * ϕ^d[j]) / σ2 / b^2 -
                                                    4 * a * d[j]^2 * ej^2 * ϕ2^(d[j] - 1) / σ2 / b^3, j)
    end

    return [fd, dd]
end

function σ_upd(ϕ::Float64, yo::AbstractVector, d::AbstractVector, itr; wf::Function = fid)
    -1 < ϕ < 1 || error("Out of bound")
    ϕ2 = ϕ^2
    s = 0.0
    l = 0
    for j in itr
        s += wf(abs2.(yo[j + 1] - (ϕ^d[j] *  yo[j])) / (1 - ϕ2^d[j]), j)
        l += wf(1, j)
    end
    
    return sqrt((1 - ϕ^2) / l * s)
end


function CA_ISARz(yo, d, st_pt, itr;
    Tlr::Float64 = 1e-8, MIte::Int = 100, pw::AbstractVector = [1.0])
    ϕ, σ = copy(st_pt)

    if pw == [1.0]
        wfun = fid
    else
        wfun(a, j) = a / pw[j]
    end
    
    z = atanh(ϕ)
    diff = 0.0
    z_grad(z) =  z_upd(z, σ, yo, d, itr; wf = wfun)
    
    for ite in 1:MIte
        σ = σ_upd(tanh(z), yo, d, itr; wf = wfun)
        zn, = NR(z_grad, z; Tlr = Tlr, MIte = MIte)
        diff = zn - z
        z = zn
        if abs(diff) < Tlr
            return [tanh(z), σ], ite
        end
    end
    error("Non convergence")
    return [tanh(z), σ], MIte
end

function CA_ISAR(yo, d, st_pt, itr;
    Tlr::Float64 = 1e-8, MIte::Int = 300, pw::AbstractVector = [1.0])
    ϕ, σ = copy(st_pt)

    if pw == [1.0]
        wfun = fid
    else
        wfun(a, j) = a / pw[j]
    end

    diff = 0.0

    ϕ_grad(ϕ) = ϕ_upd(ϕ, σ, yo, d, itr; wf = wfun)
    for ite in 1:MIte
        σ = σ_upd(ϕ, yo, d, itr; wf = wfun)
        ϕn, = NR(ϕ_grad, ϕ; Tlr = Tlr, MIte = MIte)
        diff = ϕn - ϕ
        ϕ = ϕn
        if abs(diff) < Tlr
            return [ϕ, σ], ite
        end
    end
    error("Non convergence")
    return [ϕ, σ], MIte
end


# Iterator for subsamples returns the next index being sampled
sub_itr(bits::BitArray)  = Iterators.filter(i -> Base.unsafe_bitgetindex(bits.chunks, i), Base.OneTo(length(bits)))
sub_itr(bits::AbstractArray{Bool})  = Iterators.filter(i -> bits[i], Base.OneTo(length(bits)))

# Transformation
function z_tran!(par) # FROM ϕ TO z
    par[1] = atanh(par[1])
    par[2] = abs(par[2])
    return par
end

z_tran(par) = z_tran!(copy(par)) 

function z_tran_inv!(par) # FROM z TO ϕ
    par[1] = tanh(par[1])
    par[2] = abs(par[2])
    return par
end

z_tran_inv(par) = z_tran_inv!(copy(par)) 

function z_Gra_tran(ld, Mdd, ϕ)
    s2 = 1 - ϕ^2
    tld = [ld[1] * s2, ld[2]]
    tMdd = [(s2^2 * Mdd[1, 1] - 2 * s2 * ϕ * ld[1])  (s2 * Mdd[1, 2]); (s2 * Mdd[2, 1]) Mdd[2, 2]]
    return tld, tMdd 
end

fid(x, y) = x

## Solve Optimization function
# yo is a vector of legnth m+1 where the first one is y_0
# d is a vector of length m where each value is d_{t_j - t_{j-1}}
# par is the current value of ϕ and σ
# itr is an iterator(!) over all needed indexes
# wf is the weighting function that applies to each ld and Mdd
function Op_Fun(yo, d, parθ, itr; wf::Function = fid, ret_z = true)
    ϕ, σ = parθ
    ld = zeros(2)
    Mdd = zeros(2, 2)

    
    # Speed up calculation
    ϕ2 = abs2(ϕ)
    σ2 = abs2(σ)
    a = 1 - ϕ2

    ## CAREFUL ABOUT THE INDEX OF y[j - 1]
    y = OffsetArray(yo, 0:(length(yo) - 1))
    # Reparameterization
    for j in itr
        ## Bit Array
        ej = y[j] - ϕ^d[j] * y[j-1]
        b = 1 - ϕ2^d[j]

        G2j_11 = (2 * d[j]^2 * ϕ^(2 * d[j] - 2) / b^2 - d[j] * ϕ^(2 * d[j] - 2) / b - (1 + ϕ^2) / (1 - ϕ^2)^2) +
            (ej^2 - 2 * ej * d[j] * y[j-1] * ϕ^d[j]) / σ^2 / b +
            ej^2 * ϕ^(2 * d[j]) * 2 * d[j] / σ^2 / b^2 +
            a * d[j]^2 * ϕ2^(d[j] - 1) * (2 * y[j] * ej - y[j-1]^2 + y[j]^2) / σ2 / b^2 +
            d[j] * ej * (y[j-1] - ϕ^d[j] * y[j]) * ((d[j] - 1) * (((d[j] - 1) != 0) * ϕ^(d[j] - 2)) -
                                                    (d[j] + 1) * ϕ^d[j]) / σ2 / b^2 -
                                                    4 * a * d[j]^2 * ej^2 * ϕ2^(d[j] - 1) / σ2 / b^3
        
        G2j_12 = (2 * d[j] * ej * ϕ^(d[j] - 1) * (ϕ^d[j] * y[j] - y[j - 1]) * (1 - ϕ2)) / abs(σ)^3 / b^2 -
        2 * ej^2 * ϕ / abs(σ)^3 / b

        
        G2j_22 = 1 / σ2 - 3 * ej^2 * (1 - ϕ2) / σ2^2 / b

        Mdd += wf([G2j_11 G2j_12;
                   G2j_12 G2j_22], j)
        ld += wf([d[j] * ϕ^(2 * d[j] - 1) / b - ϕ / a + ϕ * ej^2 / σ2 / b + 
            a * d[j] * ej * (ϕ^(d[j] - 1) * y[j - 1] - ϕ^(2 * d[j] - 1) * y[j]) / σ2 / b^2;
                  - 1/ σ + ej^2 * a / σ^3 / b], j)
    end
    if ret_z
        return z_Gra_tran(ld, Mdd, ϕ)
    else
        return ld, Mdd
    end
end

# For MLE

function MLE(yo, d, st_pt, itr; Tlr::Float64 = 1e-7, MIte::Int = 100)
    return  CA_ISARz(yo, d, st_pt, itr;
    Tlr = Tlr, MIte = MIte)
end

function MLE(yo, d, st_pt; Tlr::Float64 = 1e-7, MIte::Int = 100)
    return MLE(yo, d, st_pt, Base.OneTo(length(d)); Tlr = Tlr, MIte = MIte)
end

function MLE(yo, d, st_pt, plt_bits::AbstractVector{Bool}; Tlr::Float64 = 1e-7, MIte::Int = 100)
    return  MLE(yo, d, st_pt, sub_itr(plt_bits); Tlr = Tlr, MIte = MIte)
end

function MLE(yo, d, st_pt, plt_num::Int; Tlr::Float64 = 1e-7, MIte::Int = 100)
    plt_itr = sample(1 : length(d), plt_num; replace = false)
    return MLE(yo, d, st_pt, plt_itr; Tlr = Tlr, MIte = MIte)
end

function MLE(yo, d, st::Function, itr; Tlr::Float64 = 1e-7, MIte::Int = 100)
    st_pt = st(yo, d, itr)
    return  CA_ISARz(yo, d, st_pt, itr;
    Tlr = Tlr, MIte = MIte)
end

function MLE(yo, d, st::Function; Tlr::Float64 = 1e-7, MIte::Int = 100)
    st_pt = st(yo, d, 1:length(d))
    return MLE(yo, d, st_pt, Base.OneTo(length(d)); Tlr = Tlr, MIte = MIte)
end

function MLE(yo, d, st::Function, plt_bits::AbstractVector{Bool}; Tlr::Float64 = 1e-7, MIte::Int = 100)
    st_pt = st(yo, d, plt_bits)
    return  MLE(yo, d, st_pt, sub_itr(plt_bits); Tlr = Tlr, MIte = MIte)
end

function MLE(yo, d, st::Function, plt_num::Int; Tlr::Float64 = 1e-7, MIte::Int = 100)
    plt_itr = sample(1 : length(d), plt_num; replace = false)
    st_pt = st(yo, d, plt_itr)
    return MLE(yo, d, st_pt, plt_itr; Tlr = Tlr, MIte = MIte)
end

# For IPW

function IPW(yo, d, st::Function, ind, pw::AbstractVector{Float64}; Tlr::Float64 = 1e-7, MIte::Int = 100)
    st_pt = st(yo, d, ind, pw)
    return CA_ISARz(yo, d, st_pt, ind;
    Tlr = Tlr, MIte = MIte, pw = pw)
end

 # Ambiguous definition without it in case of (x, x, Function , Bool)
function IPW(yo, d, st::Function, bit_ind::AbstractVector{Bool}, pw::AbstractVector{Float64};
             Tlr::Float64 = 1e-7, MIte::Int = 100)
    return IPW(yo, d, st::Function, findall(bit_ind), pw::AbstractVector{Float64};
               Tlr = Tlr, MIte = MIte)
end

function IPW(yo, d, st_pt, ind, pw::AbstractVector{Float64}; Tlr::Float64 = 1e-7, MIte::Int = 100)
    return CA_ISARz(yo, d, st_pt, ind;
    Tlr = Tlr, MIte = MIte, pw = pw)
end


function IPW(yo, d, st_pt, bit_ind::AbstractVector{Bool}, pw::AbstractVector{Float64};
             Tlr::Float64 = 1e-7, MIte::Int = 100)
    return IPW(yo, d, st_pt, findall(bit_ind), pw::AbstractVector{Float64};
               Tlr = Tlr, MIte = MIte)
end

