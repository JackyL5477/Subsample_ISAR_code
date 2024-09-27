using Random;


gaps_gen(m::Int; set = 1:5) = rand(set, m)

function y_gen(d::AbstractVector, ϕ::Float64, σ::Float64, y0::Float64)
    m = length(d)
    y = Array{Float64}(undef, m + 1)
    
    scale = σ / sqrt(1 - abs2(ϕ))
    
    y[1] = y0    
    @inbounds for i in 1:m
        tmp = ϕ^(d[i])
        y[i + 1] =  tmp * y[i] + scale * sqrt(1 - abs2(tmp)) * randn()
    end

    return y
end

function y_gen(d::AbstractVector, ϕ::Float64, σ::Float64)
    y_gen(d, ϕ, σ, σ / sqrt(1 - abs2(ϕ)) * randn())
end


