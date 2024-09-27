using Random;
using JLD2;

include("gen.jl");
include("LS_st.jl")
include("numeric_alg_tCA.jl")
include("sub_alg_comb_H.jl")

rep = 100;

m = 10^5;
rs = collect(5000:5000:30000)
r_plt = 1000 # for Aopt and iboss
α = 0.05

est_κ(z, r) = r / sum(z);
st_pt = LS_ISAR_st # TO HELP CONVERGENCE

σs = [1.0, 10.0, 20.0];

ϕ = 0.1
time_ϕ1 = zeros(length(rs), 5, length(σs))
run_ϕ1 = zeros(Int, length(rs), length(σs))
Random.seed!(12345)
for (i3, σ) in enumerate(σs)
    for (i1, r) in enumerate(rs)
        for repi in 1:rep
            print("σ:", σ, ", r:", r, ", Rep:", repi, "   (", run_ϕ1[i1, i3], ")     \r")
            d = gaps_gen(m)
            yo = y_gen(d, ϕ, σ)
            try
                t1 = @elapsed MLE(yo, d, st_pt, r + r_plt)
                t2 = @elapsed Aopt(yo, d, r_plt, r, α; κ_fun=est_κ, st_pt=st_pt)
                t3 = @elapsed iboss(yo, d, r_plt, r; st_pt=st_pt)
                t4 = @elapsed thin(yo, d, r_plt, r, 0.7, 0.1, 1e-3; st_pt=st_pt)
                t5 = @elapsed MLE(yo, d, st_pt)
                time_ϕ1[i1, :, i3] += [t1, t2, t3, t4, t5]
                run_ϕ1[i1, i3] += 1
            catch
            end

        end
    end
end

ϕ = 0.5
time_ϕ2 = zeros(length(rs), 5, length(σs))
run_ϕ2 = zeros(Int, length(rs), length(σs))
Random.seed!(23456)
for (i3, σ) in enumerate(σs)
    for (i1, r) in enumerate(rs)
        for repi in 1:rep
            print("σ:", σ, ", r:", r, ", Rep:", repi, "   (", run_ϕ2[i1, i3], ")     \r")
            d = gaps_gen(m)
            yo = y_gen(d, ϕ, σ)

            try
                t1 = @elapsed MLE(yo, d, st_pt, r + r_plt)
                t2 = @elapsed Aopt(yo, d, r_plt, r, α; κ_fun=est_κ, st_pt=st_pt)
                t3 = @elapsed iboss(yo, d, r_plt, r; st_pt=st_pt)
                t4 = @elapsed thin(yo, d, r_plt, r, 0.7, 0.1, 1e-3; st_pt=st_pt)
                t5 = @elapsed MLE(yo, d, st_pt)
                time_ϕ2[i1, :, i3] += [t1, t2, t3, t4, t5]
                run_ϕ2[i1, i3] += 1
            catch
            end

        end
    end
end

ϕ = 0.9
time_ϕ3 = zeros(length(rs), 5, length(σs))
run_ϕ3 = zeros(Int, length(rs), length(σs))
Random.seed!(34567)
for (i3, σ) in enumerate(σs)
    for (i1, r) in enumerate(rs)
        for repi in 1:rep
            print("σ:", σ, ", r:", r, ", Rep:", repi, "   (", run_ϕ3[i1, i3], ")     \r")
            d = gaps_gen(m)
            yo = y_gen(d, ϕ, σ)

            try
                t1 = @elapsed MLE(yo, d, st_pt, r + r_plt)
                t2 = @elapsed Aopt(yo, d, r_plt, r, α; κ_fun=est_κ, st_pt=st_pt)
                t3 = @elapsed iboss(yo, d, r_plt, r; st_pt=st_pt)
                t4 = @elapsed thin(yo, d, r_plt, r, 0.7, 0.1, 1e-3; st_pt=st_pt)
                t5 = @elapsed MLE(yo, d, st_pt)
                time_ϕ3[i1, :, i3] += [t1, t2, t3, t4, t5]
                run_ϕ3[i1, i3] += 1
            catch
            end

        end
    end
end

@save "../sim_time.jld2" time_ϕ1 run_ϕ1 time_ϕ2 run_ϕ2 time_ϕ3 run_ϕ3

function find_avg(t, r; adj::Function = x -> x)
    avg = similar(t)
    for meth in 1:size(t, 2)
        avg[:, meth, :] = t[:, meth, :] ./ r
    end
    return adj.(avg)
end


using Latexify
empty_col = repeat([""], 6);

v_adj(x, dig) = x > 10^(dig-1) ? Int(round(x)) : round(x, sigdigits = dig)

avg_time_ϕ1 = find_avg(1000 .* time_ϕ1, run_ϕ1; adj = x -> v_adj(x, 3))
avg_time_ϕ2 = find_avg(1000 .* time_ϕ2, run_ϕ2; adj = x -> v_adj(x, 3))
avg_time_ϕ3 = find_avg(1000 .* time_ϕ3, run_ϕ3; adj = x -> v_adj(x, 3))

latexify([rs avg_time_ϕ1[:, :, 1];
          rs avg_time_ϕ1[:, :, 2];
          rs avg_time_ϕ1[:, :, 3]])
latexify([rs avg_time_ϕ2[:, :, 1];
          rs avg_time_ϕ2[:, :, 2];
          rs avg_time_ϕ2[:, :, 3]])
latexify([rs avg_time_ϕ3[:, :, 1];
          rs avg_time_ϕ3[:, :, 2];
          rs avg_time_ϕ3[:, :, 3]])

# σ × ϕ 
latexify([empty_col rs avg_time_ϕ1[:, :, 1] avg_time_ϕ2[:, :, 1] avg_time_ϕ3[:, :, 1]]) |> print
latexify([empty_col rs avg_time_ϕ1[:, :, 2] avg_time_ϕ2[:, :, 2] avg_time_ϕ3[:, :, 2]]) |> print
latexify([empty_col rs avg_time_ϕ1[:, :, 3] avg_time_ϕ2[:, :, 3] avg_time_ϕ3[:, :, 3]]) |> print

# ϕ × σ
latexify([empty_col rs avg_time_ϕ1[:, :, 1] avg_time_ϕ1[:, :, 2] avg_time_ϕ1[:, :, 3]]) |> print
latexify([empty_col rs avg_time_ϕ2[:, :, 1] avg_time_ϕ2[:, :, 2] avg_time_ϕ2[:, :, 3]]) |> print
latexify([empty_col rs avg_time_ϕ3[:, :, 1] avg_time_ϕ3[:, :, 2] avg_time_ϕ3[:, :, 3]]) |> print

latexify([empty_col rs avg_time_ϕ1[:, :, 1] avg_time_ϕ1[:, :, 3]]) |> print
latexify([empty_col rs avg_time_ϕ2[:, :, 1] avg_time_ϕ2[:, :, 3]]) |> print
latexify([empty_col rs avg_time_ϕ3[:, :, 1] avg_time_ϕ3[:, :, 3]]) |> print

latexify([[empty_col rs avg_time_ϕ1[:, :, 1] avg_time_ϕ1[:, :, 3]];
          [empty_col rs avg_time_ϕ2[:, :, 1] avg_time_ϕ2[:, :, 3]];
          [empty_col rs avg_time_ϕ3[:, :, 1] avg_time_ϕ3[:, :, 3]]]
        ) |> print
