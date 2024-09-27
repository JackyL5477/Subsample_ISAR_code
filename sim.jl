using Random;
using JLD2;

include("gen.jl");
include("LS_st.jl")
include("numeric_alg_tCA.jl") # tCA do full newton updating ϕ
include("sub_alg_comb_H.jl")
MSE(est, par) = sum(abs2, est .- par)#;abs2(est[2] - par[2])
f_MSE(ϕest, ϕt, y, d) = mean(abs2, yj * (ϕest^dj - ϕt^dj) for (yj, dj) in zip(y[2:end], d))
split_MSE(est, par) = Tuple(abs2.(est .- par))
full_MSE(est, par, y, d) = (f_MSE(est[1],par[1],y, d), split_MSE(est, par)...)

rep = 10;

m = 10^5;
rs = 10 .* collect(500:500:3000)
r_plt = 1000 # for Aopt and iboss
α = 0.05

est_κ(z, r) =  r / sum(z);
st_pt = LS_ISAR_st # Estimated Starting Point

σs = [1.0, 10.0, 20.0];
ϕ = 0.1; 

run_ϕ1 = zeros(Int, length(rs), length(σs));
save_ϕ1 = (forecast = zeros(length(rs), 4, length(σs)),
           ϕ = zeros(length(rs), 4, length(σs)),
           σ = zeros(length(rs), 4, length(σs)))
Random.seed!(12345)
for (i3, σ) in enumerate(σs)
    for (i1, r) in enumerate(rs)
        for repi in 1:rep
            print("σ:", σ, ", r:", r, ", Rep:", repi, "   (",run_ϕ1[i1,i3],")     \r")
            d = gaps_gen(m);
            yo = y_gen(d, ϕ, σ);
            try
                mse1_f, mse1_ϕ, mse1_σ = full_MSE(MLE(yo, d, st_pt, r + r_plt)[1], [ϕ, σ],yo, d)
                mse2_f, mse2_ϕ, mse2_σ = full_MSE(Aopt(yo, d, r_plt, r, α; κ_fun = est_κ, st_pt = st_pt)[1],[ϕ, σ],yo, d)
                mse3_f, mse3_ϕ, mse3_σ = full_MSE(iboss(yo, d, r_plt, r; st_pt = st_pt)[1], [ϕ, σ],yo, d)
                mse4_f, mse4_ϕ, mse4_σ = full_MSE(thin(yo, d, r_plt, r, 0.7, 0.1, 1e-3; st_pt = st_pt)[1], [ϕ, σ],yo, d)
                
                save_ϕ1.forecast[i1, :, i3] += [mse1_f, mse2_f, mse3_f, mse4_f]#, mse5]
                save_ϕ1.ϕ[i1, :, i3] += [mse1_ϕ, mse2_ϕ, mse3_ϕ, mse4_ϕ]#, mse5]
                save_ϕ1.σ[i1, :, i3] += [mse1_σ, mse2_σ, mse3_σ, mse4_σ]#, mse5]
                run_ϕ1[i1, i3] += 1;
            catch
            end
        end        
    end
end

ϕ = 0.5; 
run_ϕ2 = zeros(Int, length(rs), length(σs));
save_ϕ2 = (forecast = zeros(length(rs), 4, length(σs)),
           ϕ = zeros(length(rs), 4, length(σs)),
           σ = zeros(length(rs), 4, length(σs)))
Random.seed!(23456)
for (i3, σ) in enumerate(σs)
    for (i1, r) in enumerate(rs)
        for repi in 1:rep
            print("σ:", σ, ", r:", r, ", Rep:", repi, "   (",run_ϕ2[i1,i3],")     \r")
            d = gaps_gen(m);
            yo = y_gen(d, ϕ, σ);

            try
                mse1_f, mse1_ϕ, mse1_σ = full_MSE(MLE(yo, d, st_pt, r + r_plt)[1], [ϕ, σ],yo, d)
                mse2_f, mse2_ϕ, mse2_σ = full_MSE(Aopt(yo, d, r_plt, r, α; κ_fun = est_κ, st_pt = st_pt)[1],[ϕ, σ],yo, d)
                mse3_f, mse3_ϕ, mse3_σ = full_MSE(iboss(yo, d, r_plt, r; st_pt = st_pt)[1], [ϕ, σ],yo, d)
                mse4_f, mse4_ϕ, mse4_σ = full_MSE(thin(yo, d, r_plt, r, 0.7, 0.1, 1e-3; st_pt = st_pt)[1], [ϕ, σ],yo, d)
                
                save_ϕ2.forecast[i1, :, i3] += [mse1_f, mse2_f, mse3_f, mse4_f]#, mse5]
                save_ϕ2.ϕ[i1, :, i3] += [mse1_ϕ, mse2_ϕ, mse3_ϕ, mse4_ϕ]#, mse5]
                save_ϕ2.σ[i1, :, i3] += [mse1_σ, mse2_σ, mse3_σ, mse4_σ]#, mse5]
                run_ϕ2[i1, i3] += 1;
            catch

            end
            
        end        
    end
end

ϕ = 0.9; 
run_ϕ3 = zeros(Int, length(rs), length(σs));
save_ϕ3 = (forecast = zeros(length(rs), 4, length(σs)),
           ϕ = zeros(length(rs), 4, length(σs)),
           σ = zeros(length(rs), 4, length(σs)));
Random.seed!(34567)
for (i3, σ) in enumerate(σs)
    for (i1, r) in enumerate(rs)
        for repi in 1:rep
            print("σ:", σ, ", r:", r, ", Rep:", repi, "   (",run_ϕ3[i1,i3],")     \r")
            d = gaps_gen(m);
            yo = y_gen(d, ϕ, σ);

            try
                mse1_f, mse1_ϕ, mse1_σ = full_MSE(MLE(yo, d, st_pt, r + r_plt)[1], [ϕ, σ],yo, d)
                mse2_f, mse2_ϕ, mse2_σ = full_MSE(Aopt(yo, d, r_plt, r, α; κ_fun = est_κ, st_pt = st_pt)[1],[ϕ, σ],yo, d)
                mse3_f, mse3_ϕ, mse3_σ = full_MSE(iboss(yo, d, r_plt, r; st_pt = st_pt)[1], [ϕ, σ],yo, d)
                mse4_f, mse4_ϕ, mse4_σ = full_MSE(thin(yo, d, r_plt, r, 0.7, 0.1, 1e-3; st_pt = st_pt)[1], [ϕ, σ],yo, d)
                
                save_ϕ3.forecast[i1, :, i3] += [mse1_f, mse2_f, mse3_f, mse4_f]#, mse5]
                save_ϕ3.ϕ[i1, :, i3] += [mse1_ϕ, mse2_ϕ, mse3_ϕ, mse4_ϕ]#, mse5]
                save_ϕ3.σ[i1, :, i3] += [mse1_σ, mse2_σ, mse3_σ, mse4_σ]#, mse5]
                run_ϕ3[i1, i3] += 1;
            catch
            end
            
        end        
    end
end

@save "simulation.jld2" save_ϕ1 run_ϕ1 save_ϕ2 run_ϕ2 save_ϕ3 run_ϕ3

