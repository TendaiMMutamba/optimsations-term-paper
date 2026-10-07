using Plots

function plot_convergence(inst::Instance; save_path::String="convergence.png")
    lr_polyak = lagrangian_relaxation(inst; policy=POLYAK, max_iter=300)
    lr_geo = lagrangian_relaxation(inst; policy=GEOMETRIC, max_iter=300)
    lr_harm = lagrangian_relaxation(inst; policy=HARMONIC, max_iter=300)

    p = plot(size=(800, 500), dpi=150)

    plot!(p, lr_polyak.lb_history, label="Polyak LB", lw=2, color=:blue)
    plot!(p, lr_polyak.ub_history, label="Polyak UB", lw=2, color=:blue, ls=:dash)
    plot!(p, lr_geo.lb_history, label="Geometric LB", lw=2, color=:red)
    plot!(p, lr_geo.ub_history, label="Geometric UB", lw=2, color=:red, ls=:dash)
    plot!(p, lr_harm.lb_history, label="Harmonic LB", lw=2, color=:green)
    plot!(p, lr_harm.ub_history, label="Harmonic UB", lw=2, color=:green, ls=:dash)

    xlabel!(p, "Iteration")
    ylabel!(p, "Objective value")
    title!(p, "Convergence of subgradient methods")

    savefig(p, save_path)
    println("Saved convergence plot to $save_path")
    return p
end

function plot_scalability(results; save_path::String="scalability.png")
    ns = [r.N for r in results]
    mip_times = [r.mip_time for r in results]
    lr_times = [r.lr_time for r in results]

    p = plot(size=(700, 450), dpi=150)

    bar_x = repeat(ns, outer=2)
    bar_y = vcat(mip_times, lr_times)
    bar_group = vcat(fill("Direct MIP", length(ns)), fill("Lagrangian", length(ns)))

    groupedbar(ns, hcat(mip_times, lr_times),
        bar_position=:dodge, bar_width=0.35,
        label=["Direct MIP" "Lagrangian"],
        color=[:steelblue :coral],
        xlabel="Number of workloads (N)",
        ylabel="Runtime (seconds)",
        title="Runtime comparison",
        size=(700, 450), dpi=150,
        yscale=:log10)

    savefig(save_path)
    println("Saved scalability plot to $save_path")
end

function plot_multipliers(λ::Vector{Float64}, inst::Instance;
                          save_path::String="multipliers.png")
    order = sortperm(λ, rev=true)
    sorted_λ = λ[order]
    sorted_cpu = inst.d_cpu[order]

    p = plot(layout=(1, 2), size=(1000, 400), dpi=150)

    bar!(p[1], 1:inst.N, sorted_λ,
        xlabel="Workload (sorted by λ)",
        ylabel="Multiplier value (λ)",
        title="Lagrange multipliers",
        legend=false, color=:steelblue, alpha=0.8)

    scatter!(p[2], inst.d_cpu, λ,
        xlabel="CPU demand",
        ylabel="Multiplier value (λ)",
        title="Multiplier vs CPU demand",
        legend=false, color=:coral, alpha=0.6, ms=4)

    savefig(p, save_path)
    println("Saved multiplier plot to $save_path")
    return p
end

function plot_weight_sensitivity(results; save_path::String="weights.png")
    names = [r.name for r in results]
    costs = [r.cost for r in results]
    times = [r.time for r in results]
    energies = [r.energy for r in results]

    max_cost = maximum(costs)
    max_time = maximum(times)
    max_energy = maximum(energies)

    norm_cost = costs ./ max_cost
    norm_time = times ./ max_time
    norm_energy = energies ./ max_energy

    groupedbar(names, hcat(norm_cost, norm_time, norm_energy),
        bar_position=:dodge,
        label=["Cost" "Response time" "Energy"],
        color=[:steelblue :coral :seagreen],
        ylabel="Normalised value",
        title="Objective trade-offs under different weight configurations",
        xrotation=20,
        size=(800, 450), dpi=150)

    savefig(save_path)
    println("Saved weight sensitivity plot to $save_path")
end

function generate_all_plots(; output_dir::String=".")
    println("\nGenerating plots...")

    inst = medium_instance(seed=42)

    plot_convergence(inst; save_path=joinpath(output_dir, "fig1_convergence.png"))

    println("  Running scalability experiment for plot...")
    scale_results = run_experiment5(verbose=false)
    plot_scalability(scale_results; save_path=joinpath(output_dir, "fig2_scalability.png"))

    λ, _, inst4 = run_experiment4(verbose=false)
    plot_multipliers(λ, inst4; save_path=joinpath(output_dir, "fig3_multipliers.png"))

    weight_results = run_experiment6(verbose=false)
    plot_weight_sensitivity(weight_results; save_path=joinpath(output_dir, "fig4_weights.png"))

    println("\nAll plots saved to $output_dir")
end
