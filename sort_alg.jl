### From dual select to opt_index methods
using LinearAlgebra
using LoopVectorization

function fsum(fun::Function, a) # Fast sum
    s = 0
    @turbo for i ∈ eachindex(a)# in 1:length(a)
        s += fun(a[i])
    end
    return s   
end

function fsum(a) # Fast sum
    s = 0
    @turbo for i ∈ eachindex(a)# in 1:length(a)
        s += a[i]
    end
    return s   
end

function fsum(fun::Function, a, index) # Fast sum
    s = 0
    @turbo for i ∈ index
        s += fun(a[i])
    end
    return s   
end


function dotprod(a, b) # Fast sum of product
    s = 0.0
    @turbo for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    return s
end

function kat!(a::Vector, inds::AbstractVector{Bool})
    p::Int = 1
    for (q, i) in enumerate(inds)
        Base._copy_item!(a, p, q)
        p += i
    end
    Base._deleteend!(a, length(a) - p + 1)
    return a
end

function dat!(a::Vector, inds::AbstractVector{Bool})
    p::Int = 1
    for (q, i) in enumerate(inds)
        Base._copy_item!(a, p, q)
        p += !i
    end
    Base._deleteend!(a, length(a) - p + 1)
    return a
end

# Need at least one 0 in inds
function kat_unsafe!(a::Vector, inds::AbstractVector{Bool})
    p::Int = 1
    @turbo for (q, i) in enumerate(inds)
        a[p] = a[q]
        p += i
    end
    Base._deleteend!(a, length(a) - p + 1)
    return a
end

# Need at least one 1 in inds
function dat_unsafe!(a::Vector, inds::AbstractVector{Bool})
    p::Int = 1
    @turbo for (q, i) in enumerate(inds)
        a[p] = a[q]
        p += !i
    end
    Base._deleteend!(a, length(a) - p + 1)
    return a
end

# Same as idx .= xi <= f but saves memory
function assign!(idx::BitVector, x::AbstractVector{T}, f::T) where {T}
    resize!(idx, length(x))
    @turbo for (i, xi) in enumerate(x)
        idx[i] = xi <= f
    end
end

function assign!(idx::BitVector, x::AbstractVector{T}, Fun::Function) where {T}
    resize!(idx, length(x))
    @turbo for (i, xi) in enumerate(x)
        idx[i] = Fun(xi)
    end
end

    

function sat_tmp!(a::Vector{T}, b::Vector{T}, inds::AbstractVector{Bool}; s::Int = sum(inds)) where {T}
    resize!(b, s + 1)
    pa::Int = 1
    pb::Int = 1
    for (q, i) in enumerate(inds)
        Base._copy_item!(a, pa, q)
        @inbounds b[pb] = a[q]
        pa += !i
        pb += i
    end
    # pb == (s+1) || error("unequal: sat!")
    Base._deleteend!(a, length(a) - pa + 1)
    Base._deleteend!(b, 1)
    return b
end

## The 
function opt_select(xo::AbstractVector, r::Int)
    n = length(xo)
    n < r && error("Insufficient vector size.")

    comp = BitArray(undef, n)
    f = xo[1]
    assign!(comp, xo, f)

    # Copy the result to a new memory location
    if sum(comp) > r
        x = xo[comp]
        Base._deletebeg!(x, 1)
    elseif sum(comp) < r
        r = r - sum(comp)
        x = xo[.!comp]
    else
        return f
    end

    # Looping
    while true
        f = popfirst!(x)
        assign!(comp, x, f)

        s = sum(comp)
        if s > r - 1
            kat!(x, comp)
        elseif s < r - 1
            r -= s + 1
            dat!(x, comp)
        else
            return f
        end
    end
end

function opt_select!(x::Vector, r::Int)
    n = length(x)
    n < r && error("Insufficient vector size.")

    comp = BitArray(undef, n)

    # Looping
    while true
        f = popfirst!(x)
        assign!(comp, x, f)

        s = sum(comp)
        if s > r - 1
            kat!(x, comp)
        elseif s < r - 1
            r -= s + 1
            dat!(x, comp)
        else
            return f
        end
    end
end

## TRIAL 1

function kless!(a::Vector, v::Real) # keep less = delete more
    p::Int = 1
    for q in Base.OneTo(length(a))
        if @inbounds a[q] < v
            @inbounds a[p] = a[q]
            p += 1
        end
    end
    Base._deleteend!(a, length(a) - p + 1)
    return a
end

function dless!(a::Vector, v::Real) # delete less = keep more
    p::Int = 1
    for q in Base.OneTo(length(a))
        if @inbounds a[q] >= v
            @inbounds a[p] = a[q]
            p += 1
        end
    end
    Base._deleteend!(a, length(a) - p + 1)
    return a
end

function keepwhen!(a::Vector, op::Function) # keep less = delete more
    p::Int = 1
    for q in Base.OneTo(length(a))
        @inbounds a[p] = a[q]
        if @inbounds op(a[q])
            p += 1
        end
    end
    Base._deleteend!(a, length(a) - p + 1)
    return a
end


## Option 1

# improves performance but is hard to read
function tfsum(a, κ_inv)
    s1::Float64 = 0
    s2::Int64 = 0
    @turbo for i ∈ eachindex(a)
        ai_ind = a[i] < κ_inv
        s1 += ai_ind * a[i]
        s2 += !ai_ind
    end
    return s1, s2
end

function find_κ(PI::AbstractVector{Float64}, r::Int)
    if length(PI) < r
        error("Subsample size greater than full data size.")
    end
    # Choose a value based on the first element in the list
    κ_inv = fsum(PI) / r
    sl, sg = tfsum(PI, κ_inv)
    s = sg + sl / κ_inv

    while !isapprox(s, r)
        κ_inv = 1 / (1/κ_inv - (s - r)  / sl)
        sl, sg = tfsum(PI, κ_inv)
        s = sg  + sl / κ_inv
        end
    return 1 / κ_inv
end

function copyless!(dst::AbstractArray{T}, a::AbstractArray{T}, v) where T
    p = 1
    @inbounds for i ∈ eachindex(a)
        dst[p] = a[i]
        p += a[i] < v
    end
end

function find_κ_loc(PI::AbstractVector{Float64}, r::Int)
    if length(PI) < r
        error("Subsample size greater than full data size.")
    end

    κ_inv = fsum(PI) / r
    sl, sg = tfsum(PI, κ_inv)
    s = sg + sl / κ_inv

    if isapprox(s, r)
        return 1 / κ_inv
    end
    keep_pi = Array{Float64}(undef, length(PI) - sg)
    copyless!(keep_pi, PI, κ_inv)
    sg_t = 0.0
    
    while !isapprox(s, r)
        κ_inv = 1 / (1/κ_inv - (s - r)  / sl)
        sg_t += sg
        sl, sg = tfsum(keep_pi, κ_inv)
        s = (sg + sg_t)  + sl / κ_inv
        kless!(keep_pi, κ_inv)
    end
    return 1/κ_inv
end

