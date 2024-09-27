using JLD2
using Plots
using LaTeXStrings

# @load "sim1.jld2" save_ϕ1 err_ϕ1 save_ϕ2 err_ϕ2 save_ϕ3 err_ϕ3
# @save "sim1.jld2" save_ϕ1 run_ϕ1 save_ϕ2 run_ϕ2 save_ϕ3 run_ϕ3

# @load "sim2.jld2" run_ϕ3

plot(rs, save_ϕ1[:, :, 1] ./ (run_ϕ1[:, 1]),
     label=["Uniform" "Aopt" "iboss" "thin"],
     markershape = :circle,
     ms = 2,
     msw = 0)
plot!(grid=false,
      size=(400, 400),
      frame=:box,
      legend=:topright,
      ylabel="MSE",
      xlabel="Subsample size (r)",
      yticks = 0:0.002:0.02)
savefig("σ1.png")

plot(rs, save_ϕ1[: ,: , 2] ./ (run_ϕ1[:, 2]),
     label=["Uniform" "Aopt" "iboss" "thin"],
     markershape = :circle,
     ms = 2,
     msw = 0)
plot!(grid=false,
      size=(400, 400),
      frame=:box,
      legend=:topright,
      ylabel="MSE",
      xlabel="Subsample size (r)",
      yticks = 0:0.02:0.2)
savefig("σ10.png")

plot(rs, save_ϕ1[: ,: , 3] ./ (run_ϕ1[:, 3]),
     label=["Uniform" "Aopt" "iboss" "thin"],
     markershape = :circle,
     ms = 2,
     msw = 0)
plot!(grid=false,
      size=(400, 400),
      frame=:box,
      legend=:topright,
      ylabel="MSE",
      xlabel="Subsample size (r)",
      yticks = 0:0.1:0.5)
  savefig("σ20.png")

let i = 1
    plot(rs, save_ϕ1[:, :, i] ./ (run_ϕ1[:, i]),
        label=["Uniform" "Aopt" "iboss" "thin" "thin2"],
        markershape=:circle,
        ms=2,
        msw=0)
    plot!(grid=false,
        size=(300, 300),
        frame=:box,
        legend=:topright,
        yrotation=90,
          yticks=5,
          yformatter=:plain,
        ylabel="MSE",
        xlabel="Subsample size (r)")
end

# let i = 1
function format_yticks(y_data)
    # Find the range of the data
    min_y = minimum(y_data)
    max_y = maximum(y_data)

    # Determine the order of magnitude
    magnitude = floor(Int, log10(max_y))
    scale_factor = exp10(magnitude)

    # Scale the data for the ticks
    scaled_min_y = min_y / scale_factor
    scaled_max_y = max_y / scale_factor

    # Generate tick positions and labels
    tick_positions = range(scaled_min_y, scaled_max_y, length=5) * scale_factor
    tick_labels = string.(round.(tick_positions / scale_factor, digits=1))

    return (tick_positions, tick_labels), magnitude
end

fe(str) = eval(Meta.parse(str))

for case = 1:3
    for i = 1:3
        let res = fe("save_ϕ$(case)"), ct = fe("run_ϕ$(case)"), fign = "fig/$(case)_$i.pdf"
            y_tick, y_mag = format_yticks(res[:, :, i] ./ (ct[:, i]))
            plot(rs ./ 1000, res[:, :, i] ./ (ct[:, i]),
                label=["Uniform" "Aopt" "iboss" "thin" "thin2"],
                markershape=:circle,
                ms=2,
                msw=0)
            plot!(grid=false,
                size=(300, 300),
                frame=:box,
                legend=:topright,
                yrotation=90,
                yticks=y_tick,
                xformatter=:plain,
                ylabel="MSE ( " * L"\times 10^{%$(y_mag)}" * ")",
                xlabel="Subsample Size r ( " * L"\times 10^{3}" * ")")
            savefig(fign)
        end
    end
end

let i = 1
    plot(rs, save_ϕ1[:, :, i] ./ (run_ϕ1[:, i]),
        label=["Uniform" "Aopt" "iboss" "thin" "thin2"],
        markershape=:circle,
        ms=2,
        msw=0)
    plot!(grid=false,
        size=(300, 300),
        frame=:box,
        legend=:topright,
        yrotation=90,
        yformatter=:scientific,
        xformatter=:plain,
        ylabel="MSE",
        xlabel="Subsample size (r)")
end
# let i = 1
# plot(rs, sum(x -> ismissing(x) ? 0 : x, save_ϕ3, dims = 4)[:, i, :] ./ (run_ϕ3[:, :, i]),
#      label=["Uniform" "Aopt" "iboss" "thin" "thin2"],
#      markershape = :circle,
#      ms = 2,
#      msw = 0)
# plot!(grid=false,
#       size=(300, 300),
#       frame=:box,
#       legend=:topright,
#       ylabel="MSE",
#       xlabel="Subsample size (r)")
# end
