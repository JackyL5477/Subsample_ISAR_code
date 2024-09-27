using Statistics
using LoopVectorization

function ISAR_LS(yo, d, dv, bit_itr::AbstractVector{Bool}; wf::Function = fid)
    num = denom = 0
    @turbo for (j,δ) in enumerate(BitVector(bit_itr))
        num += δ * wf((d[j] == dv) * yo[j] * yo[j + 1], j)
        denom += δ * wf((d[j] == dv) * abs2(yo[j]), j)
    end
    val = num/ denom
    # return copysign(abs(val) ^ (1/dv), val)
    ret = iseven(dv) && signbit(val) ? missing : copysign(abs(val) ^ (1/dv), val)
    return ret
end

function ISAR_LS(yo, d, dv, itr; wf::Function = fid)
    num = denom = 0
    @turbo for itr_idx in eachindex(itr)
        j = itr[itr_idx]
        num += wf(yo[j] * yo[j + 1] * (d[j] == dv), j)
        denom += wf(abs2(yo[j]) * (d[j] == dv), j)
    end
    val = num/ denom
    # return copysign(abs(val) ^ (1/dv), val)
    ret = iseven(dv) && signbit(val) ? missing : copysign(abs(val) ^ (1/dv), val)
    return ret
end

function ISAR_LSMSE(ϕ::Float64, yo::AbstractVector, d::AbstractVector, bit_itr::AbstractVector{Bool}; wf::Function = fid)
    -1 < ϕ < 1 || error("Out of bound")
    ϕ2 = ϕ^2
    l = s = 0.0
    # @turbo
    for (j,δ) in enumerate(BitVector(bit_itr))
        # ϕj = (1 - 2 * isodd(j) * signbit(ϕ)) * abs(ϕ)^j
        # s += δ * wf(abs2(yo[j + 1] - (ϕj *  yo[j])) / (1 - ϕ2^d[j]), j)
        s += δ * wf((abs2(yo[j + 1] - (ϕ^d[j] *  yo[j])) / (1 - ϕ2^d[j])), j)
        l += wf(δ, j)
    end
    
    return sqrt((1 - ϕ^2) / l * s)
end

function ISAR_LSMSE(ϕ::Float64, yo::AbstractVector, d::AbstractVector, itr; wf::Function = fid)
    -1 < ϕ < 1 || error("Out of bound")
    ϕ2 = ϕ^2
    l = s = 0.0
    # @turbo
    for itr_idx in 1:length(itr)
        j = itr[itr_idx]
        # ϕj = (1 - 2 * isodd(j) * signbit(ϕ)) * abs(ϕ)^j
        # s += wf(abs2(yo[j + 1] - (ϕj *  yo[j])) / (1 - ϕ2^d[j]), j)
        s += wf(abs2(yo[j + 1] - (ϕ^d[j] *  yo[j])) / (1 - ϕ2^d[j]), j)
        l += wf(1, j)
    end
    
    return sqrt((1 - ϕ^2) / l * s)
end


function LS_ISAR_st(yo, d, ind)
    ds = unique(d[ind])
    ϕs = Array{Union{Missing, Float64}}(undef, length(ds))

    for (i, di) in enumerate(ds)
        ϕs[i]  = ISAR_LS(yo, d, di, ind)
    end
    # ϕs = isodd.(ds) .&& signbit.(ϕs)
    ϕ = mean(skipmissing(ϕs))
    σ = ISAR_LSMSE(ϕ, yo, d, ind)

    return [ϕ, σ]
end

function LS_ISAR_st(yo, d, ind, pw)
    ds = unique(d[ind])
    ϕs = Array{Union{Missing, Float64}}(undef, length(ds))
    ipw(a, j) = a / pw[j]

    for (i, di) in enumerate(ds)
        ϕs[i]  = ISAR_LS(yo, d, di, ind; wf = ipw)
    end
    # ϕs = isodd.(ds) .&& signbit.(ϕs)
    ϕ = mean(skipmissing(ϕs))
    σ = ISAR_LSMSE(ϕ, yo, d, ind; wf = ipw)

    return [ϕ, σ]
end
